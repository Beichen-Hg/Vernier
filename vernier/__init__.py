"""Vernier: Single-Pass Joint Evidence Selection for Long-Term Conversational Memory."""

from vernier.losses import compute_vernier_loss
from vernier.modeling import (
    BidirectionalSetHead,
    VernierSelector,
    build_selector_input,
    collate_selector_batch,
)
from vernier.packing import (
    knapsack_pack,
    render_evidence_packet,
    render_inline_time_grounding,
)
from vernier.prompts import (
    SELECTOR_SYSTEM_PROMPT,
    build_reader_messages,
)

__all__ = [
    "BidirectionalSetHead",
    "VernierSelector",
    "build_selector_input",
    "collate_selector_batch",
    "compute_vernier_loss",
    "knapsack_pack",
    "render_evidence_packet",
    "render_inline_time_grounding",
    "SELECTOR_SYSTEM_PROMPT",
    "build_reader_messages",
]
