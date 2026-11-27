"""Evaluate Vernier calibration (ECE, Brier, NLL, AUROC, AUPRC), Turn All-Gold EM,
and reverse-order permutation stability over Top-32 benchmark pools.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vernier import VernierSelector, build_selector_input, collate_selector_batch, knapsack_pack


def compute_calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p <= hi) if b == n_bins - 1 else (p >= lo) & (p < hi)
        if np.any(m):
            ece += (m.sum() / len(y)) * abs(float(y[m].mean()) - float(p[m].mean()))
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    auroc = float(roc_auc_score(y, p)) * 100.0 if len(np.unique(y)) > 1 else 50.0
    auprc = float(average_precision_score(y, p)) * 100.0 if len(np.unique(y)) > 1 else 0.0
    gold_mask = y > 0.5
    gold_lt05 = float(np.mean(p[gold_mask] < 0.05)) * 100.0 if np.any(gold_mask) else 0.0
    return {
        "ece15": round(ece, 4),
        "brier": round(brier, 4),
        "nll": round(nll, 4),
        "auroc": round(auroc, 2),
        "auprc": round(auprc, 2),
        "gold_lt05_pct": round(gold_lt05, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Vernier on a pre-built benchmark Top-32 pool pickle.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--pools-pkl", type=str, required=True)
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = VernierSelector.from_pretrained(args.checkpoint, base_model_name_or_path=args.base_model, device=args.device)
    recs = pickle.load(open(args.pools_pkl, "rb"))

    def infer(reversed_order: bool = False) -> list[np.ndarray]:
        out_probs: list[np.ndarray] = []
        for st in range(0, len(recs), args.batch_size):
            sub = recs[st : st + args.batch_size]
            exs, perms = [], []
            for r in sub:
                n_c = len(r["top32"])
                perm = list(range(n_c - 1, -1, -1)) if reversed_order else list(range(n_c))
                perms.append(perm)
                exs.append(build_selector_input(tok, r, perm=perm, max_cand_tokens=72))
            bt = collate_selector_batch(exs, pad_id=pad_id, device=args.device, max_k=32)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                oo = model(
                    bt["input_ids"], bt["attention_mask"], bt["cand_pos"],
                    bt["suff_pos"], bt["cand_feats"], bt["cand_mask"],
                )
                pb = oo["probs"].float().cpu().numpy()
            for bi, perm in enumerate(perms):
                inv = np.empty(len(perm), dtype=np.int64)
                for new_i, old_i in enumerate(perm):
                    inv[old_i] = new_i
                out_probs.append(pb[bi, inv])
        return out_probs

    p_norm = infer(reversed_order=False)
    p_rev = infer(reversed_order=True)

    y_all, p_all = [], []
    em_hits, rev_em_hits, tok_used, sp_vals = [], [], [], []
    for r, pn, pr in zip(recs, p_norm, p_rev):
        top = r["top32"]
        em = r.get("eval_meta") or {}
        gsids = set(em.get("gold_sids") or [])
        gsess = set(em.get("gold_sess_ids") or [])
        use_sess = len(gsids) == 0 and len(gsess) > 0
        gset = gsess if use_sess else gsids

        y_q = [1 if ((c["sess_id"] in gsess) if use_sess else (c["source_id"] in gsids)) else 0 for c in top]
        y_all.extend(y_q)
        p_all.extend(pn.tolist())

        toks = [int(c["tok_len"]) for c in top]
        sel_n = knapsack_pack(pn, toks, budget=args.budget, tau=args.tau, k_min=args.k_min)
        sel_r = knapsack_pack(pr, toks, budget=args.budget, tau=args.tau, k_min=args.k_min)
        tok_used.append(sum(toks[i] for i in sel_n))

        rho = spearmanr(pn, pr).statistic
        if np.isfinite(rho):
            sp_vals.append(float(rho))

        if not em.get("is_abstention", False) and gset:
            key = "sess_id" if use_sess else "source_id"
            em_hits.append(int(gset.issubset({top[i][key] for i in sel_n})))
            rev_em_hits.append(int(gset.issubset({top[i][key] for i in sel_r})))

    metrics = {
        "n_queries": len(recs),
        "n_answerable": len(em_hits),
        "calibration": compute_calibration_metrics(np.asarray(y_all), np.asarray(p_all)),
        "all_gold_em_pct": round(100.0 * float(np.mean(em_hits)), 2) if em_hits else 0.0,
        "rev_all_gold_em_pct": round(100.0 * float(np.mean(rev_em_hits)), 2) if rev_em_hits else 0.0,
        "mean_selected_tokens": round(float(np.mean(tok_used)), 1),
        "rev_spearman": round(float(np.mean(sp_vals)), 4) if sp_vals else 1.0,
    }
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
