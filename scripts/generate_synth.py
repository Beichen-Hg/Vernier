"""Capability-driven synthetic longitudinal dialogue generator (Appendix A.3 & Box E4).

Reads API credentials strictly from environment variables (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.request

DOMAINS = [
    "friends_social_chat",
    "user_assistant_log",
    "team_project_sync",
    "service_advisor_followup",
    "research_study_collab",
    "family_household_group",
]

FIRST_NAMES = [
    "Elena", "Marcus", "Priya", "Liam", "Sofia", "Noah", "Hana", "Devon", "Claire", "Mateo",
    "Nadia", "Owen", "Tessa", "Julian", "Mei", "Caleb", "Amara", "Victor", "Iris", "Samir",
    "Nora", "Felix", "Maya", "Ethan", "Zoe", "Adrian", "Leila", "Gavin", "Chloe", "Rafael",
    "Simone", "Kieran", "Yuki", "Isaac", "Vera", "Logan", "Ananya", "Miles", "Daria", "Elliot",
    "Soren", "Bianca", "Tariq", "Celeste", "Damon", "Freya", "Gideon", "Helena", "Ivan", "Juno",
    "Kendra", "Lucian", "Maren", "Nikolai", "Orla", "Phoebe", "Quentin", "Rohan", "Sylvie", "Tobias",
]

STORY_ARCS = [
    ("pottery glaze kiln experiments", "adopting a senior rescue beagle", "training for an autumn half-marathon", "balcony herb drip system"),
    ("migrating payment ledger to Rust", "SOC2 Type-II compliance audit", "reducing p99 search tail latency", "on-call incident rotation"),
    ("sourdough boule hydration tuning", "rooftop solar inverter installation", "planning 10th anniversary rail trip", "garage woodworking bench"),
    ("rotator cuff post-op physical therapy", "ferritin & iron nutrition protocol", "home office split-keyboard ergonomics", "sleep schedule stabilization"),
    ("sparse MoE pruning submission", "distributed systems course lab grading", "summer research lab internship interviews", "cluster GPU quota allocation"),
    ("toddler bilingual daycare waitlist", "attic skylight flashing repair", "refinancing fixed-rate mortgage", "summer lake cabin booking"),
    ("custom mechanical switch lubing", "indoor hydroponic chili peppers", "freelance motion-graphics retainer", "film camera darkroom prints"),
    ("documentary festival sound mix", "indie newsletter sponsorship tiers", "restoring 1980s analog polysynth", "local trail cleanup group"),
    ("Kubernetes spot-node autoscaling", "checkout funnel A/B experiment", "enterprise SSO SAML rollout", "intern capstone mentorship"),
    ("asthma controller inhaler step-down", "marathon electrolyte gel testing", "vestibular balance drill routine", "gluten-free meal batching"),
    ("solid-state battery XRD characterization", "NSF grant budget re-justification", "conference workshop keynote prep", "visiting postdoc housing"),
    ("eldercare physical therapist schedule", "backyard cedar pergola build", "household ETF savings rebalance", "electric vehicle L2 charger permit"),
    ("amateur deep-sky telescope guiding", "community chamber choir rehearsals", "fostering twin shelter kittens", "sourdough pizza oven weekend"),
    ("PostgreSQL time-series partitioning", "Python async worker memory leak", "Q3 engineering offsite in Porto", "API rate-limiter redesign"),
    ("wedding string quartet contract", "scenic alpine honeymoon passes", "allergen-safe bakery tasting", "guest shuttle schedule"),
    ("cryo-EM beamtime sample vitrification", "fellowship specific aims rewrite", "linear algebra recitation quizzes", "open-source benchmark release"),
    ("condo roof assessment appeal", "529 education fund allocation", "extended family reunion lodge", "minivan winter tire swap"),
    ("indie roguelike Steam demo polish", "pixel-art boss animation commission", "wrist tendonitis rehab exercises", "discord playtest feedback"),
    ("warehouse RF scanner firmware patch", "packaging supplier lead-time negotiation", "quarterly cycle-count audit", "cold-chain sensor alerts"),
    ("cello conservatory audition piece", "violin luthier soundpost adjustment", "summer chamber music residency", "sheet music tablet pedal setup"),
]

SYSTEM_PROMPT = (
    "You are an expert longitudinal conversational memory dataset architect. "
    "Return strictly valid JSON matching the requested schema. "
    "Never leak internal IDs (such as S0, Ch1, t3) inside dialogue utterances, questions, or answers."
)


def make_blueprint(hist_idx: int) -> dict:
    rng = random.Random(20261118 + hist_idx * 104729)
    domain = DOMAINS[hist_idx % len(DOMAINS)]
    b1 = STORY_ARCS[(hist_idx // len(DOMAINS) + rng.randint(0, len(STORY_ARCS) - 1)) % len(STORY_ARCS)]
    b2 = STORY_ARCS[(hist_idx * 5 + 11) % len(STORY_ARCS)]
    arcs = [b1[0], b1[1], b1[2], b2[3]]
    names = rng.sample(FIRST_NAMES, 3)

    if domain == "user_assistant_log":
        speakers = ["User", "Assistant"]
    elif domain == "service_advisor_followup":
        speakers = [names[0], f"{rng.choice(['Advisor', 'Coach', 'Clinician'])}_{names[1]}"]
    elif domain in ("team_project_sync", "research_study_collab", "family_household_group"):
        speakers = names[: (3 if hist_idx % 2 == 0 else 2)]
    else:
        speakers = names[:2]

    cur = dt.date(rng.choice([2023, 2024, 2025]), rng.randint(1, 5), rng.randint(2, 22))
    chapter_dates = []
    for _ in range(4):
        c_dates = []
        for _ in range(8):
            c_dates.append(cur.strftime("%Y-%m-%d"))
            cur += dt.timedelta(days=rng.randint(3, 9))
        chapter_dates.append(c_dates)
        cur += dt.timedelta(days=rng.randint(6, 14))
    q_ref_date = (cur + dt.timedelta(days=rng.randint(4, 12))).strftime("%Y-%m-%d")

    return {
        "conv_id": f"syn_{hist_idx:04d}",
        "split": "val" if hist_idx < 30 else "train",
        "domain": domain,
        "speakers": speakers,
        "arcs": arcs,
        "chapter_dates": chapter_dates,
        "q_ref_date": q_ref_date,
    }


def build_chapter_prompt(bp: dict, ch_idx: int) -> str:
    sp_str = ", ".join(bp["speakers"])
    dates_str = ", ".join(f"S{i}={d}" for i, d in enumerate(bp["chapter_dates"][ch_idx]))
    a0, a1, a2, a3 = bp["arcs"]
    stage_desc = [
        "Chapter 1/4: Initial plans, baseline numbers/choices, and early activities.",
        "Chapter 2/4: Ongoing progress, mid-course complication, and relative-date events.",
        "Chapter 3/4: Explicit state update/pivot superseding an earlier Chapter 1--2 plan/number.",
        "Chapter 4/4: Final outcomes, reflection via pronouns/near-synonyms, and settled preferences.",
    ][ch_idx]
    return f"""Generate Chapter {ch_idx + 1}/4 (8 chronological sessions, 7--9 turns/session) of longitudinal conversation `{bp['conv_id']}` (domain: `{bp['domain']}`), plus 2 intra-chapter QAs.

BLUEPRINT:
- Speakers: [{sp_str}]
- Role: {stage_desc}
- Session Dates: {dates_str}
- Evolving Storylines: (A) {a0}; (B) {a1}; (C) {a2}; (D) {a3}

TURN-LENGTH & SUPERVISION REQUIREMENTS:
1. Mix ~70% natural 12--42 word turns (including intra-turn coreference, 2-turn elliptical Q&A, and relative dates) with ~30% structured 110--210 word multi-sentence turns (1--2 buried facts amid detachable tips/boilerplate) so length cannot act as a shortcut.
2. Include realistic phatic acknowledgments and rejected-plan distractors.
3. Return strict JSON with keys `chapter_summary`, `sessions`, and `intra_qas`."""


def call_chat_json(api_key: str, base_url: str, model: str, user_prompt: str, max_retries: int = 5) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.55,
        "max_tokens": 8192,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            txt = (res["choices"][0]["message"].get("content") or "").strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S).strip()
            return json.loads(txt)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(15.0, 2.0 ** attempt))
    raise RuntimeError("Unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate capability-driven synthetic longitudinal dialogues.")
    ap.add_argument("--num-histories", type=int, default=2, help="Number of 4-chapter histories to synthesize.")
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash-1017")
    ap.add_argument("--output", type=str, default="synthetic_chapters.jsonl")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY in your environment before running generation.")

    out_path = Path(args.output)
    with out_path.open("a", encoding="utf-8") as fout:
        for h_idx in range(args.num_histories):
            bp = make_blueprint(h_idx)
            for ch in range(4):
                data = call_chat_json(api_key, base_url, args.model, build_chapter_prompt(bp, ch))
                fout.write(json.dumps({"blueprint": bp, "chapter": ch, "output": data}, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[Generated] {bp['conv_id']} chapter {ch + 1}/4", flush=True)


if __name__ == "__main__":
    main()
