"""Single-GPU training script for Vernier (reproducing Table A1 hyperparameters)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
from scipy.stats import rankdata
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vernier import (
    BidirectionalSetHead,
    VernierSelector,
    build_selector_input,
    collate_selector_batch,
    compute_vernier_loss,
    knapsack_pack,
)


def spearman_corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 1.0
    ra = rankdata(a).astype(np.float64)
    rb = rankdata(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Vernier LoRA + BidirectionalSetHead on pre-built Top-32 pools.")
    ap.add_argument("--pools-pkl", type=str, required=True, help="Path to synthetic Top-32 pool pickle (train/val splits).")
    ap.add_argument("--output-dir", type=str, required=True, help="Directory to save best lora_adapter/ and head.pt.")
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-1.7B", help="Base backbone model ID or path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr-lora", type=float, default=1.5e-4)
    ap.add_argument("--lr-head", type=float, default=4.0e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-cand-tokens", type=int, default=72)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = out_dir / "lora_adapter"
    head_path = out_dir / "head.pt"

    tok = AutoTokenizer.from_pretrained(args.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    pools = pickle.load(open(args.pools_pkl, "rb"))
    train_recs = [r for r in pools if r["split"] == "train" and len(r["top32"]) >= 2]
    val_recs = [r for r in pools if r["split"] == "val" and len(r["top32"]) >= 2]
    print(f"[Train] Loaded train={len(train_recs)}, val={len(val_recs)} pools.", flush=True)

    base_enc = AutoModel.from_pretrained(args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa")
    base_enc.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    enc = get_peft_model(base_enc, lora_cfg)
    head = BidirectionalSetHead(hidden_size=int(base_enc.config.hidden_size), nhead=16, num_layers=2, dropout=0.1)
    model = VernierSelector(enc, head).to(args.device)

    steps_per_ep = math.ceil(len(train_recs) / (args.batch_size * args.grad_accum))
    tot_steps = steps_per_ep * args.epochs
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in enc.parameters() if p.requires_grad], "lr": args.lr_lora, "weight_decay": args.weight_decay},
            {"params": list(head.parameters()), "lr": args.lr_head, "weight_decay": args.weight_decay},
        ]
    )
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=max(10, int(0.08 * tot_steps)),
        num_training_steps=tot_steps,
    )

    best_val_score = -1e9
    rng = random.Random(args.seed + 999)
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_recs)
        opt.zero_grad(set_to_none=True)
        acc_stats: dict[str, float] = defaultdict(float)
        acc_cnt = 0
        opt_step = 0

        for b_idx, st in enumerate(range(0, len(train_recs), args.batch_size)):
            batch_recs = train_recs[st : st + args.batch_size]
            ex1_list, perms1 = [], []
            for r in batch_recs:
                p1 = list(range(len(r["top32"])))
                rng.shuffle(p1)
                perms1.append(p1)
                ex1_list.append(build_selector_input(tok, r, perm=p1, max_cand_tokens=args.max_cand_tokens))
            bt1 = collate_selector_batch(ex1_list, pad_id=pad_id, device=args.device, max_k=32)

            do_perm = (b_idx % 3 == 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out1 = model(
                    bt1["input_ids"], bt1["attention_mask"], bt1["cand_pos"],
                    bt1["suff_pos"], bt1["cand_feats"], bt1["cand_mask"],
                )
                out2_aligned = None
                if do_perm:
                    ex2_list, inv_maps = [], []
                    for r, p1 in zip(batch_recs, perms1):
                        p2 = list(range(len(r["top32"])))
                        rng.shuffle(p2)
                        ex2_list.append(build_selector_input(tok, r, perm=p2, max_cand_tokens=args.max_cand_tokens))
                        pos_in_p2 = {orig_i: idx2 for idx2, orig_i in enumerate(p2)}
                        inv_maps.append([pos_in_p2[orig_i] for orig_i in p1])
                    bt2 = collate_selector_batch(ex2_list, pad_id=pad_id, device=args.device, max_k=32)
                    out2 = model(
                        bt2["input_ids"], bt2["attention_mask"], bt2["cand_pos"],
                        bt2["suff_pos"], bt2["cand_feats"], bt2["cand_mask"],
                    )
                    th2_al = torch.zeros_like(out1["theta"])
                    for bi, imap in enumerate(inv_maps):
                        idx_t = torch.tensor(imap, dtype=torch.long, device=args.device)
                        th2_al[bi, : len(imap)] = out2["theta"][bi, idx_t]
                    out2_aligned = {"theta": th2_al}

                loss, st_d = compute_vernier_loss(out1, bt1, out2_aligned=out2_aligned)
                (loss / args.grad_accum).backward()

            for k, v in st_d.items():
                acc_stats[k] += v
            acc_cnt += 1

            if (acc_cnt % args.grad_accum == 0) or (st + args.batch_size >= len(train_recs)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 50 == 0 or opt_step == steps_per_ep:
                    avg = {k: round(acc_stats[k] / max(1, acc_cnt), 4) for k in ("total", "bce", "rank", "perm", "prem")}
                    print(f"[ep={ep} step={opt_step}/{steps_per_ep} {time.perf_counter()-t0:.0f}s] {avg}", flush=True)

        # Synthetic validation: maximize All-Gold EM@750 + 10 * RevSp
        model.eval()
        val_em = val_n = 0
        val_sp: list[float] = []
        for r in val_recs:
            em_d = r["eval_meta"]
            if em_d.get("is_abstention") or not em_d.get("gold_sids"):
                continue
            gs = set(em_d["gold_sids"])
            n_c = len(r["top32"])
            ex_n = build_selector_input(tok, r, perm=list(range(n_c)), max_cand_tokens=args.max_cand_tokens)
            ex_r = build_selector_input(tok, r, perm=list(range(n_c - 1, -1, -1)), max_cand_tokens=args.max_cand_tokens)
            bt = collate_selector_batch([ex_n, ex_r], pad_id=pad_id, device=args.device, max_k=32)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                oo = model(
                    bt["input_ids"], bt["attention_mask"], bt["cand_pos"],
                    bt["suff_pos"], bt["cand_feats"], bt["cand_mask"],
                )
                th_n = oo["theta"][0, :n_c].float().cpu().numpy()
                th_r = oo["theta"][1, :n_c].float().cpu().numpy()[::-1]
            val_sp.append(spearman_corr(th_n.tolist(), th_r.tolist()))
            probs_n = 1.0 / (1.0 + np.exp(-np.clip(th_n, -20.0, 20.0)))
            toks = [c["tok_len"] for c in r["top32"]]
            sel = knapsack_pack(probs_n, toks, budget=750, tau=0.0, k_min=1)
            val_em += int(gs.issubset({r["top32"][i]["source_id"] for i in sel}))
            val_n += 1

        em_pct = 100.0 * val_em / max(1, val_n)
        rev_sp = float(np.mean(val_sp)) if val_sp else 0.0
        val_score = em_pct + 10.0 * rev_sp
        print(f"[VAL ep={ep}] All-Gold EM@750={em_pct:.2f}% | RevSp={rev_sp:.4f} | score={val_score:.2f}", flush=True)
        if val_score > best_val_score:
            best_val_score = val_score
            enc.save_pretrained(str(lora_dir))
            torch.save(head.state_dict(), head_path)
            print(f"  -> Saved best checkpoint to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
