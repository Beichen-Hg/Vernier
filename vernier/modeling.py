"""Vernier single-pass bidirectional set scoring architecture (Section 3.1, Eqs. 1, 2, 4).

Matches the released Vernier checkpoints (`Qwen/Qwen3-1.7B` rank-32 LoRA + 2-layer
position-free Pre-LN bidirectional set head) bit-for-bit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from vernier.prompts import SELECTOR_SYSTEM_PROMPT


def build_selector_input(
    tokenizer: Any,
    record: dict[str, Any],
    perm: Optional[list[int]] = None,
    max_cand_tokens: int = 72,
) -> dict[str, Any]:
    """Construct unindexed single-prefill inputs and order-invariant pool features (Eqs. 1, 4).

    Args:
        tokenizer: Qwen3 tokenizer.
        record: Pool dictionary containing `question`, optional `question_date`,
            `top32` (list of candidate turn dicts), and optional `eval_meta`.
        perm: Permutation of candidate indices (defaults to natural order).
        max_cand_tokens: Per-candidate token cap (default: 72).

    Returns:
        Dictionary of token IDs, readout positions, order-invariant 4-dim features `f_i`,
        and optional supervision labels.
    """
    top = record["top32"]
    n_c = len(top) if perm is None else len(perm)
    if perm is None:
        perm = list(range(len(top)))
    cands = [top[i] for i in perm]

    q_date = str(record.get("question_date") or "").strip()
    q_hdr = (
        f"[Reference Date: {q_date}]\nQuestion: {record['question']}"
        if q_date
        else f"Question: {record['question']}"
    )
    prefix_str = (
        f"<|im_start|>system\n{SELECTOR_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{q_hdr}\n\nCandidates:\n"
    )
    input_ids: list[int] = tokenizer.encode(prefix_str, add_special_tokens=False)

    # Readout sentinels: single-token `</C>` and chat-end `<S>`
    marker_ids = tokenizer.encode(" <|vision_pad|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        marker_ids = tokenizer.encode(" >>", add_special_tokens=False)[-1:]
    marker_id = marker_ids[0]
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    suff_ids = tokenizer.encode("\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    # Four order-invariant pool statistics (Eq. 4)
    scores = [float(c["rerank_score"]) for c in cands]
    s_mean = float(np.mean(scores)) if scores else 0.0
    s_std = max(0.5, float(np.std(scores))) if len(scores) > 1 else 1.0
    s_max = max(scores) if scores else 0.0
    sess_ords = [int(c.get("sess_ordinal", 0)) for c in cands]
    so_min = min(sess_ords) if sess_ords else 0
    so_span = max(1, (max(sess_ords) - so_min)) if sess_ords else 1

    # Unindexed candidate formatting (zero numeric rank IDs)
    body_strs = [
        f"[{c['date_hdr']} | {c['speaker']}] {c['verbatim_render']}"
        for c in cands
    ]
    all_body_ids = tokenizer(body_strs, add_special_tokens=False)["input_ids"] if body_strs else []

    cand_pos: list[int] = []
    feats = np.zeros((n_c, 4), dtype=np.float32)
    for idx, (c, raw_ids) in enumerate(zip(cands, all_body_ids)):
        body_ids = raw_ids[:max_cand_tokens]
        input_ids.extend(body_ids)
        input_ids.append(marker_id)
        cand_pos.append(len(input_ids) - 1)
        input_ids.extend(nl_ids)

        sc = float(c["rerank_score"])
        feats[idx, 0] = (sc - s_mean) / s_std
        feats[idx, 1] = np.clip((sc - s_max) / 4.0, -4.0, 0.0)
        feats[idx, 2] = math.log1p(max(1, int(c["tok_len"]))) / 5.0
        feats[idx, 3] = (int(c.get("sess_ordinal", 0)) - so_min) / float(so_span)

    input_ids.extend(suff_ids)
    suff_pos = len(input_ids) - 1

    # Supervision labels (populated during training/evaluation only)
    em = record.get("eval_meta") or {}
    gold_sids = set(em.get("gold_sids") or [])
    gold_sess = set(em.get("gold_sess_ids") or [])
    contrast_sids = set(em.get("contrast_sids") or [])
    use_sess = len(gold_sids) == 0 and len(gold_sess) > 0

    y_keep = [
        1.0 if ((c["sess_id"] in gold_sess) if use_sess else (c["source_id"] in gold_sids)) else 0.0
        for c in cands
    ]
    y_contrast = [1.0 if (c["source_id"] in contrast_sids) else 0.0 for c in cands]
    y_abs = 1.0 if bool(em.get("is_abstention", False)) else 0.0

    return {
        "input_ids": input_ids,
        "cand_pos": cand_pos,
        "suff_pos": suff_pos,
        "cand_feats": feats,
        "y_keep": y_keep,
        "y_contrast": y_contrast,
        "y_abs": y_abs,
        "perm": perm,
    }


def collate_selector_batch(
    examples: list[dict[str, Any]],
    pad_id: int,
    device: Union[str, torch.device],
    max_k: int = 32,
) -> dict[str, torch.Tensor]:
    """Pad variable-length prefill prompts to a multiple of 8 for tensor-core efficiency."""
    bsz = len(examples)
    max_len = max(len(e["input_ids"]) for e in examples)
    max_len = ((max_len + 7) // 8) * 8

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    cand_pos = torch.zeros((bsz, max_k), dtype=torch.long)
    suff_pos = torch.zeros((bsz,), dtype=torch.long)
    cand_feats = torch.zeros((bsz, max_k, 4), dtype=torch.float32)
    cand_mask = torch.zeros((bsz, max_k), dtype=torch.bool)
    y_keep = torch.zeros((bsz, max_k), dtype=torch.float32)
    y_contrast = torch.zeros((bsz, max_k), dtype=torch.float32)
    y_abs = torch.zeros((bsz,), dtype=torch.float32)

    for b, e in enumerate(examples):
        seq_len = len(e["input_ids"])
        input_ids[b, :seq_len] = torch.tensor(e["input_ids"], dtype=torch.long)
        attn_mask[b, :seq_len] = 1
        k = min(len(e["cand_pos"]), max_k)
        if k > 0:
            cand_pos[b, :k] = torch.tensor(e["cand_pos"][:k], dtype=torch.long)
            cand_feats[b, :k] = torch.from_numpy(e["cand_feats"][:k])
            cand_mask[b, :k] = True
            y_keep[b, :k] = torch.tensor(e["y_keep"][:k], dtype=torch.float32)
            y_contrast[b, :k] = torch.tensor(e["y_contrast"][:k], dtype=torch.float32)
        suff_pos[b] = int(e["suff_pos"])
        y_abs[b] = float(e["y_abs"])

    return {
        "input_ids": input_ids.to(device, non_blocking=True),
        "attention_mask": attn_mask.to(device, non_blocking=True),
        "cand_pos": cand_pos.to(device, non_blocking=True),
        "suff_pos": suff_pos.to(device, non_blocking=True),
        "cand_feats": cand_feats.to(device, non_blocking=True),
        "cand_mask": cand_mask.to(device, non_blocking=True),
        "y_keep": y_keep.to(device, non_blocking=True),
        "y_contrast": y_contrast.to(device, non_blocking=True),
        "y_abs": y_abs.to(device, non_blocking=True),
    }


class BidirectionalSetHead(nn.Module):
    """Residual pool-stats MLP + 2-layer position-free Pre-LN bidirectional encoder + calibrated head.

    Preserves exact state-dict parameter names (`feat_proj`, `in_norm`, `encoder`,
    `out_norm`, `theta_head`, `log_tau`, `prem_head`) for direct checkpoint loading.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        nhead: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(4, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )
        self.in_norm = nn.LayerNorm(hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.theta_head = nn.Linear(hidden_size, 1)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.prem_head = nn.Linear(hidden_size, 1)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_tau), min=0.5, max=3.0)

    def forward(
        self,
        h_suff: torch.Tensor,
        h_cands: torch.Tensor,
        cand_feats: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz = h_cands.shape[0]
        h_c = h_cands + self.feat_proj(cand_feats)
        seq = self.in_norm(torch.cat([h_suff, h_c], dim=1))
        suff_valid = torch.ones((bsz, 1), dtype=torch.bool, device=cand_mask.device)
        key_padding_mask = ~torch.cat([suff_valid, cand_mask], dim=1)
        z = self.out_norm(self.encoder(seq, src_key_padding_mask=key_padding_mask))

        temp = self.temperature
        theta = self.theta_head(z[:, 1:, :]).squeeze(-1) / temp
        prem_logit = self.prem_head(z[:, 0, :]).squeeze(-1)
        return {
            "theta": theta,
            "probs": torch.sigmoid(theta),
            "prem_logit": prem_logit,
            "h_cands_out": z[:, 1:, :],
            "h_suff_out": z[:, 0, :],
        }


class VernierSelector(nn.Module):
    """Intact zero-decode causal prefill backbone (`Qwen3-1.7B` + LoRA) + `BidirectionalSetHead`."""

    def __init__(self, encoder: nn.Module, head: BidirectionalSetHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cand_pos: torch.Tensor,
        suff_pos: torch.Tensor,
        cand_feats: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hs = out.last_hidden_state
        bsz, m_k = cand_pos.shape
        b_idx = torch.arange(bsz, device=hs.device)
        h_suff = hs[b_idx, suff_pos].unsqueeze(1).float()
        h_cands = hs[b_idx.unsqueeze(1).expand(bsz, m_k), cand_pos].float()
        return self.head(h_suff, h_cands, cand_feats, cand_mask)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        base_model_name_or_path: str = "Qwen/Qwen3-1.7B",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        merge_lora: bool = True,
        local_files_only: bool = False,
    ) -> "VernierSelector":
        """Load a trained Vernier checkpoint (`lora_adapter/` + `head.pt`) for inference."""
        from peft import PeftModel
        from transformers import AutoModel

        ckpt = Path(checkpoint_dir)
        base_enc = AutoModel.from_pretrained(
            base_model_name_or_path,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=local_files_only,
        )
        peft_enc = PeftModel.from_pretrained(base_enc, str(ckpt / "lora_adapter"))
        enc = peft_enc.merge_and_unload() if merge_lora else peft_enc

        hdim = int(base_enc.config.hidden_size)
        head = BidirectionalSetHead(hidden_size=hdim, nhead=16, num_layers=2, dropout=0.1)
        head_state = torch.load(ckpt / "head.pt", map_location="cpu", weights_only=True)
        head.load_state_dict(head_state)
        return cls(enc, head).to(device).eval()
