"""Minimal inference demo for Vernier (single-pass bidirectional scoring + exact 0/1 knapsack packing)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vernier import (
    VernierSelector,
    build_selector_input,
    collate_selector_batch,
    knapsack_pack,
    render_evidence_packet,
)


EXAMPLE_RECORD = {
    "question": "How many days passed from starting ukulele lessons to taking my guitar to the tech?",
    "question_date": "2023-04-01",
    "top32": [
        {
            "t_pos": 12,
            "source_id": "turn_g1",
            "sess_id": "sess_feb01",
            "sess_ordinal": 4,
            "iso_date": "2023-02-01",
            "date_hdr": "2023-02-01",
            "speaker": "user",
            "rerank_score": -8.5,
            "tok_len": 28,
            "verbatim_render": "user: I just started ukulele lessons with Rachel this morning!",
        },
        {
            "t_pos": 18,
            "source_id": "turn_t1",
            "sess_id": "sess_feb01",
            "sess_ordinal": 4,
            "iso_date": "2023-02-01",
            "date_hdr": "2023-02-01",
            "speaker": "user",
            "rerank_score": -6.1,
            "tok_len": 30,
            "verbatim_render": "user: I'll probably take my guitar to the tech sometime soon.",
        },
        {
            "t_pos": 95,
            "source_id": "turn_g2",
            "sess_id": "sess_feb25",
            "sess_ordinal": 14,
            "iso_date": "2023-02-25",
            "date_hdr": "2023-02-25",
            "speaker": "user",
            "rerank_score": -7.8,
            "tok_len": 26,
            "verbatim_render": "user: Decided to take my guitar to the tech today for a full setup.",
        },
        {
            "t_pos": 140,
            "source_id": "turn_noise",
            "sess_id": "sess_mar10",
            "sess_ordinal": 20,
            "iso_date": "2023-03-10",
            "date_hdr": "2023-03-10",
            "speaker": "user",
            "rerank_score": -8.2,
            "tok_len": 32,
            "verbatim_render": "user: My neighbor bought a new acoustic amplifier for weekend jams.",
        },
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Vernier single-pass evidence selection on an example pool.")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to Vernier checkpoint dir (lora_adapter/ + head.pt).")
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-1.7B", help="Base Qwen3-1.7B model path or HF ID.")
    ap.add_argument("--budget", type=int, default=1000, help="Token budget B (default: 1000).")
    ap.add_argument("--tau", type=float, default=0.05, help="Admission threshold tau (default: 0.05).")
    ap.add_argument("--k-min", type=int, default=2, help="Fallback minimum turns k_min (default: 2).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    model = VernierSelector.from_pretrained(
        checkpoint_dir=args.checkpoint,
        base_model_name_or_path=args.base_model,
        device=args.device,
        dtype=dtype,
    )

    ex = build_selector_input(tok, EXAMPLE_RECORD, max_cand_tokens=72)
    batch = collate_selector_batch([ex], pad_id=pad_id, device=args.device, max_k=len(EXAMPLE_RECORD["top32"]))

    with torch.inference_mode():
        out = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["cand_pos"],
            batch["suff_pos"],
            batch["cand_feats"],
            batch["cand_mask"],
        )
        probs = out["probs"][0, : len(EXAMPLE_RECORD["top32"])].float().cpu().numpy()
        p_abs = float(torch.sigmoid(out["prem_logit"][0]).item())

    tok_lens = [c["tok_len"] for c in EXAMPLE_RECORD["top32"]]
    selected_idx = knapsack_pack(probs, tok_lens, budget=args.budget, tau=args.tau, k_min=args.k_min)
    selected_cands = [EXAMPLE_RECORD["top32"][i] for i in selected_idx]

    print(f"Question: {EXAMPLE_RECORD['question']}")
    print(f"Unanswerable-query probability: {p_abs:.4f}\n")
    print("Candidate Admission Scores:")
    for i, (cand, prob) in enumerate(zip(EXAMPLE_RECORD["top32"], probs)):
        mark = "KEEP" if i in selected_idx else "DROP"
        print(f"  [{mark}] p={prob:.4f} | {cand['source_id']} | {cand['verbatim_render']}")

    print("\nRendered Chronological Evidence Packet:")
    print(render_evidence_packet(selected_cands, question_date=EXAMPLE_RECORD["question_date"]))


if __name__ == "__main__":
    main()
