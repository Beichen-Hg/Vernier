"""Budget-constrained exact 0/1 knapsack packing (Section 3.2, Eq. 3, Algorithm 1)
and deterministic chronological evidence rendering.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Optional, Sequence

import numpy as np

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
WORD_NUM_MAP = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "couple": 2, "couple of": 2,
}

_DATE_RE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_INLINE_TAG_RE = re.compile(
    r"\s*\[(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{4}|\d{4}|week of [^\]]+|the week before [^\]]+)\]"
)
_REL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(the\s+day\s+before\s+yesterday)\b(?!\s*\[)", re.I), "dby"),
    (re.compile(r"\b(the\s+day\s+after\s+tomorrow)\b(?!\s*\[)", re.I), "dat"),
    (re.compile(r"\b(yesterday)\b(?!\s*\[)", re.I), "yest"),
    (re.compile(r"\b(tomorrow)\b(?!\s*\[)", re.I), "tom"),
    (re.compile(r"\b(last\s+week)\b(?!\s*\[)", re.I), "last_week"),
    (re.compile(r"\b(next\s+week)\b(?!\s*\[)", re.I), "next_week"),
    (re.compile(r"\b(last\s+month)\b(?!\s*\[)", re.I), "last_month"),
    (re.compile(r"\b(next\s+month)\b(?!\s*\[)", re.I), "next_month"),
    (re.compile(r"\b(last\s+year)\b(?!\s*\[)", re.I), "last_year"),
    (re.compile(r"\b(next\s+year)\b(?!\s*\[)", re.I), "next_year"),
    (
        re.compile(
            r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a\s+couple\s+of)\s+days?\s+ago)\b(?!\s*\[)",
            re.I,
        ),
        "n_days_ago",
    ),
    (
        re.compile(
            r"\b((?:\d+|one|two|three|four|five|six|seven|eight|a\s+couple\s+of)\s+weeks?\s+ago)\b(?!\s*\[)",
            re.I,
        ),
        "n_weeks_ago",
    ),
    (
        re.compile(
            r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a\s+couple\s+of)\s+months?\s+ago)\b(?!\s*\[)",
            re.I,
        ),
        "n_months_ago",
    ),
    (
        re.compile(
            r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a\s+couple\s+of)\s+years?\s+ago)\b(?!\s*\[)",
            re.I,
        ),
        "n_years_ago",
    ),
    (
        re.compile(
            r"\b((?:last|this\s+past)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b(?!\s*\[)",
            re.I,
        ),
        "last_weekday",
    ),
    (
        re.compile(
            r"\b(next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b(?!\s*\[)",
            re.I,
        ),
        "next_weekday",
    ),
]


def knapsack_pack(
    probs: Sequence[float] | np.ndarray,
    tok_lens: Sequence[int],
    budget: int = 1000,
    tau: float = 0.05,
    k_min: int = 2,
) -> list[int]:
    """Exact 0/1 knapsack dynamic programming for expected-utility packing (Eq. 3, Algorithm 1).

    Maximizes sum_{i in S} (p_i - tau) subject to sum_{i in S} ell_i <= budget.
    If fewer than `k_min` turns satisfy p_i > tau, bypasses DP and greedily selects
    up to `k_min` highest-scoring turns within `budget` (keeping at least argmax p_i).
    """
    n = len(probs)
    if n == 0:
        return []
    p = np.asarray(probs, dtype=np.float64)
    c = [max(1, int(x)) for x in tok_lens]

    pos = [i for i in range(n) if p[i] > tau]
    if len(pos) < k_min:
        order = list(np.argsort(-p))
        sel: list[int] = []
        used = 0
        for i in order:
            if len(sel) >= k_min:
                break
            if used + c[i] <= budget or not sel:
                sel.append(int(i))
                used += c[i]
        return sel

    dp = np.full(budget + 1, -1e18, dtype=np.float64)
    dp[0] = 0.0
    keep = np.zeros((len(pos), budget + 1), dtype=bool)
    for k, i in enumerate(pos):
        ci, ui = c[i], float(p[i] - tau)
        if ci <= budget:
            cand = dp[: budget + 1 - ci] + ui
            better = cand > dp[ci:]
            dp[ci:] = np.where(better, cand, dp[ci:])
            keep[k, ci:] = better

    cur_b = int(np.argmax(dp))
    sel = []
    for k in range(len(pos) - 1, -1, -1):
        if keep[k, cur_b]:
            sel.append(pos[k])
            cur_b -= c[pos[k]]
    return sel if sel else [int(np.argmax(p))]


def parse_calendar_date(raw: str) -> Optional[dt.date]:
    m = _DATE_RE.search(str(raw or ""))
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def strip_inline_time_tags(text: str) -> str:
    return _INLINE_TAG_RE.sub("", text)


def _format_natural_date(d: dt.date) -> str:
    return f"{d.day} {MONTH_NAMES[d.month]} {d.year}"


def _shift_months(d: dt.date, delta_months: int) -> tuple[int, int]:
    total_m = (d.year * 12 + (d.month - 1)) + delta_months
    return total_m // 12, (total_m % 12) + 1


def render_inline_time_grounding(text: str, iso_date: str) -> str:
    """Resolve English relative time expressions inline against an anchor `YYYY-MM-DD` date."""
    if not text or not iso_date:
        return text
    anchor = parse_calendar_date(iso_date)
    if anchor is None:
        return text

    out = text
    for pat, kind in _REL_PATTERNS:
        def _repl(match: re.Match[str]) -> str:
            phrase = match.group(1)
            low = phrase.lower()
            tag = ""
            if kind == "dby":
                tag = _format_natural_date(anchor - dt.timedelta(days=2))
            elif kind == "dat":
                tag = _format_natural_date(anchor + dt.timedelta(days=2))
            elif kind == "yest":
                tag = _format_natural_date(anchor - dt.timedelta(days=1))
            elif kind == "tom":
                tag = _format_natural_date(anchor + dt.timedelta(days=1))
            elif kind == "last_week":
                tag = f"the week before {_format_natural_date(anchor)}"
            elif kind == "next_week":
                tag = f"the week after {_format_natural_date(anchor)}"
            elif kind == "last_month":
                y, m = _shift_months(anchor, -1)
                tag = f"{MONTH_NAMES[m]} {y}"
            elif kind == "next_month":
                y, m = _shift_months(anchor, +1)
                tag = f"{MONTH_NAMES[m]} {y}"
            elif kind == "last_year":
                tag = f"{anchor.year - 1}"
            elif kind == "next_year":
                tag = f"{anchor.year + 1}"
            elif kind in ("n_days_ago", "n_weeks_ago", "n_months_ago", "n_years_ago"):
                num_part = re.sub(r"\s+(?:days?|weeks?|months?|years?)\s+ago$", "", low).strip()
                num_part = re.sub(r"^a\s+", "", num_part).strip()
                n = int(num_part) if num_part.isdigit() else WORD_NUM_MAP.get(num_part)
                if n is not None and 1 <= n <= 60:
                    if kind == "n_days_ago":
                        tag = _format_natural_date(anchor - dt.timedelta(days=n))
                    elif kind == "n_weeks_ago":
                        tag = f"around {_format_natural_date(anchor - dt.timedelta(days=7 * n))}"
                    elif kind == "n_months_ago":
                        y, m = _shift_months(anchor, -n)
                        tag = f"{MONTH_NAMES[m]} {y}"
                    elif kind == "n_years_ago":
                        tag = f"{anchor.year - n}"
            elif kind == "last_weekday":
                target_wd = WEEKDAY_MAP.get(low.split()[-1])
                if target_wd is not None:
                    diff = (anchor.weekday() - target_wd) % 7 or 7
                    tag = _format_natural_date(anchor - dt.timedelta(days=diff))
            elif kind == "next_weekday":
                target_wd = WEEKDAY_MAP.get(low.split()[-1])
                if target_wd is not None:
                    diff = (target_wd - anchor.weekday()) % 7 or 7
                    tag = _format_natural_date(anchor + dt.timedelta(days=diff))
            return f"{phrase} [{tag}]" if tag else phrase

        out = pat.sub(_repl, out)
    return out


def render_evidence_packet(
    selected_cands: list[dict[str, Any]],
    question_date: str = "",
    include_calendar_offsets: bool = True,
) -> str:
    """Render selected verbatim turns chronologically grouped by session (Box E1)."""
    if not selected_cands:
        return "(No relevant conversation evidence found.)"

    ordered = sorted(selected_cands, key=lambda c: (int(c.get("sess_ordinal", 0)), int(c["t_pos"])))
    groups: list[tuple[tuple[int, str], list[dict[str, Any]]]] = []
    for c in ordered:
        s_key = (int(c.get("sess_ordinal", 0)), str(c.get("date_hdr") or ""))
        if not groups or groups[-1][0] != s_key:
            groups.append((s_key, [c]))
        else:
            groups[-1][1].append(c)

    d_ref = parse_calendar_date(question_date) if include_calendar_offsets else None
    prev_d_sess: Optional[dt.date] = None
    blocks: list[str] = []

    for (_, date_hdr), group_cands in groups:
        c0 = group_cands[0]
        if include_calendar_offsets:
            d_sess = parse_calendar_date(str(c0.get("iso_date") or date_hdr))
            hdr_parts = [date_hdr] if date_hdr else ["Unknown Date"]
            if d_ref is not None and d_sess is not None:
                delta_ref = (d_ref - d_sess).days
                if delta_ref > 0:
                    hdr_parts.append(f"Offset: {delta_ref} days ({delta_ref / 7.0:.1f} weeks) before Reference Date")
                elif delta_ref == 0:
                    hdr_parts.append("Offset: same day as Reference Date")
                else:
                    hdr_parts.append(f"Offset: {-delta_ref} days after Reference Date")
            if prev_d_sess is not None and d_sess is not None and d_sess > prev_d_sess:
                dp = (d_sess - prev_d_sess).days
                if dp >= 7:
                    hdr_parts.append(f"+{dp} days ({dp / 7.0:.1f} weeks) after previous session")
                else:
                    hdr_parts.append(f"+{dp} days after previous session")
            if d_sess is not None:
                prev_d_sess = d_sess
            hdr_str = " | ".join(hdr_parts)
        else:
            hdr_str = date_hdr or "Unknown Date"

        cur_lines = [f"=== Conversation Session [{hdr_str}] ==="]
        for x in group_cands:
            vr = str(x["verbatim_render"])
            if not include_calendar_offsets:
                vr = strip_inline_time_tags(vr)
            cur_lines.append(vr)
        blocks.append("\n".join(cur_lines))

    return "\n\n".join(blocks)
