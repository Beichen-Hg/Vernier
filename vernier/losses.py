"""Four-term joint training objective for Vernier (Section 3.1, Appendix A.1 Eqs. 5--7)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_vernier_loss(
    out1: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    out2_aligned: Optional[dict[str, torch.Tensor]] = None,
    lambda_rank: float = 0.55,
    lambda_perm: float = 0.50,
    lambda_prem: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute L = L_BCE + 0.55 L_rank + 0.50 L_perm + 0.10 L_prem.

    Args:
        out1: Forward pass outputs on batch permutation pi_1.
        batch: Collated batch containing `cand_mask`, `y_keep`, `y_contrast`, `y_abs`.
        out2_aligned: Optional forward pass outputs on a second shuffle pi_2,
            realigned back to pi_1 index order (enabled every third micro-batch).
        lambda_rank: Weight for pairwise softplus margin loss (default: 0.55).
        lambda_perm: Weight for dual-permutation score consistency MSE (default: 0.50).
        lambda_prem: Weight for global unanswerable-query BCE (default: 0.10).

    Returns:
        `(total_loss, stats_dict)`
    """
    theta = out1["theta"]
    prem_logit = out1["prem_logit"]
    cmask = batch["cand_mask"]
    yk = batch["y_keep"]
    yc = batch["y_contrast"]
    y_abs = batch["y_abs"]
    bsz = theta.shape[0]

    # 1. Weighted label-smoothed BCE (Eq. 5)
    y_smooth = yk * 0.94 + 0.03
    node_weights = 1.0 + 1.5 * yk + 0.75 * yc
    raw_bce = F.binary_cross_entropy_with_logits(theta, y_smooth, reduction="none") * node_weights
    loss_bce = (raw_bce * cmask.float()).sum() / torch.clamp(cmask.float().sum(), min=1.0)

    # 2. Softplus pairwise margin over answerable pools (Eq. 6, margin m = 1.5)
    rank_terms = []
    for b in range(bsz):
        vm = cmask[b]
        pos_idx = torch.where(vm & (yk[b] > 0.5))[0]
        neg_idx = torch.where(vm & (yk[b] < 0.5))[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            diff = theta[b, pos_idx].unsqueeze(1) - theta[b, neg_idx].unsqueeze(0)
            w_neg = 1.0 + 0.75 * yc[b, neg_idx].unsqueeze(0)
            rank_terms.append((F.softplus(1.5 - diff) * w_neg).mean())
    loss_rank = torch.stack(rank_terms).mean() if rank_terms else theta.sum() * 0.0

    # 3. Dual-permutation consistency MSE (Eq. 7)
    loss_perm = theta.sum() * 0.0
    if out2_aligned is not None:
        p1 = torch.sigmoid(theta)
        p2 = torch.sigmoid(out2_aligned["theta"])
        loss_perm = ((p1 - p2).pow(2) * cmask.float()).sum() / torch.clamp(cmask.float().sum(), min=1.0)

    # 4. Global unanswerable-query supervision
    loss_prem = F.binary_cross_entropy_with_logits(prem_logit, y_abs)

    total = loss_bce + lambda_rank * loss_rank + lambda_perm * loss_perm + lambda_prem * loss_prem
    stats = {
        "total": float(total.detach().cpu()),
        "bce": float(loss_bce.detach().cpu()),
        "rank": float(loss_rank.detach().cpu()),
        "perm": float(loss_perm.detach().cpu()),
        "prem": float(loss_prem.detach().cpu()),
    }
    return total, stats
