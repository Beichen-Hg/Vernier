"""Verbatim symmetric prompt templates across selector, readers, and judges (Appendix E.1, Boxes E1--E3)."""

from __future__ import annotations

SELECTOR_SYSTEM_PROMPT = (
    "You evaluate candidate conversational memory turns jointly as an evidence set. "
    "Identify which turns are needed to answer the question and which pairs are complementary "
    "(state updates, multi-hop bridges, multi-session counts) versus redundant distractors."
)

LOCOMO_READER_SYSTEM = (
    "You are a precise conversational memory Q&A assistant. "
    "Answer concisely based strictly on the provided conversation sessions. "
    "Output only the final answer, with no reasoning, JSON, or wrapper text."
)

LONGMEMEVAL_INSTRUCTION = (
    "Answer the question using the provided conversation evidence. When combining, comparing, or "
    "calculating information is necessary, you may include brief relevant evidence and calculations. "
    "Check that the facts you use satisfy the conditions in the question and that your conclusion "
    "agrees with those facts. If the evidence is insufficient, say what is missing rather than guessing."
)

UTILMEM_READER_SYSTEM = (
    "You are an expert personal assistant answering complex user requests by synthesizing "
    "distributed facts, constraints, preferences, and updates across multi-session conversation history."
)

LOCOMO_CAT5_JUDGE_PROMPT = """You are evaluating an AI assistant on an UNANSWERABLE / ADVERSARIAL conversational memory question.
The question asks about an event, fact, or detail that was NEVER mentioned in the conversation history (or contains a false premise).

Question: {question}
Known False / Adversarial Trap Answer: {adversarial_answer}
Model Response: {generated_answer}

Rules:
- Label CORRECT if the model clearly abstains, states the information is not mentioned / unknown / missing / insufficient, or refuses to fabricate an answer.
- Label WRONG if the model hallucinates an answer, guesses, or echoes the adversarial trap answer as fact.

Return ONLY a JSON object with keys "reason" (one short sentence) and "label" ("CORRECT" or "WRONG")."""

UTILMEM_RUBRIC_JUDGE_PROMPT = """You are an expert, strictly impartial evaluator grading an AI assistant's response to a complex multi-session conversational memory request.
You are given the user's request, the COMPLETE ground-truth reference sessions containing all relevant user history, and the assistant's response.

User Request:
{question}

Ground-Truth Reference Sessions (verbatim user history):
{gold_sessions}

Assistant Response:
{generated_answer}

Score each dimension from 0 to 5 (integers):
1. "grounding" (0-5): Are all claims strictly supported by the ground-truth sessions with zero hallucinations or unsupported assumptions?
2. "coverage" (0-5): How completely does the response incorporate the distributed facts, constraints, preferences, and timeline updates across the reference sessions?
3. "synthesis" (0-5): How coherently and helpfully are the multi-session details integrated and reconciled into a directly actionable response?
4. "distractor_resistance" (0-5): Does the response stay cleanly focused on the user's actual request and ground-truth facts without injecting unrelated/irrelevant distractor details?

Also compute:
- "overall_score": integer 0 to 100 reflecting overall evidence utilization quality.
- "pass": integer 1 if the response successfully utilizes the key ground-truth evidence to fulfill the request accurately (overall_score >= 60 and grounding >= 3 and coverage >= 3), else 0.

Return ONLY a JSON object with exact keys: "grounding", "coverage", "synthesis", "distractor_resistance", "overall_score", "pass", "reason"."""


def build_reader_messages(
    dataset: str,
    question: str,
    question_date: str,
    evidence: str,
) -> list[dict[str, str]]:
    """Build identical symmetric chat messages for any compared selector on a benchmark."""
    if dataset in ("longmemeval_s", "longmemeval_m"):
        if question_date:
            user_str = (
                f"{LONGMEMEVAL_INSTRUCTION}\n\nConversation evidence:\n{evidence}\n\n"
                f"Question reference date: {question_date}\nQuestion: {question}\nAnswer:"
            )
        else:
            user_str = (
                f"{LONGMEMEVAL_INSTRUCTION}\n\nConversation evidence:\n{evidence}\n\n"
                f"Question: {question}\nAnswer:"
            )
        return [{"role": "user", "content": user_str}]

    if dataset == "locomo_1986":
        user_str = (
            "Answer the question concisely based strictly on the conversation sessions below.\n"
            "Guidelines:\n"
            "1. State the direct factual answer concisely (names, items, places, or short phrases; separate multiple items with commas).\n"
            "2. When asked when something happened or given relative time expressions (such as 'yesterday', 'last week', 'next month', or 'last year'), "
            "always output the resolved concrete calendar date, period, month, or year (using the session dates and inline date tags, e.g., '7 May 2023', 'June 2023', '2022')—never answer with bare relative words.\n"
            "3. If the provided conversation sessions do not contain enough information to answer the question (or ask about something never mentioned), reply strictly with: 'Not mentioned in the conversation.'\n"
            "4. Return ONLY the concise final answer with no explanation.\n\n"
            f"Context:\n{evidence}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return [
            {"role": "system", "content": LOCOMO_READER_SYSTEM},
            {"role": "user", "content": user_str},
        ]

    if dataset == "utilmem":
        user_str = (
            "Answer the user's request thoroughly and accurately using ONLY the relevant facts, timelines, "
            "preferences, and constraints supported by the conversation evidence below.\n"
            "Guidelines:\n"
            "1. Synthesize all relevant details across multiple sessions coherently and reconcile any updates chronologically.\n"
            "2. Stay strictly grounded in the evidence—ignore unrelated distractor topics and do not hallucinate unsupported facts.\n"
            "3. Provide a clear, well-structured, directly actionable response.\n\n"
            f"Conversation Evidence:\n{evidence}\n\n"
            f"User Request: {question}\n\n"
            "Response:"
        )
        return [
            {"role": "system", "content": UTILMEM_READER_SYSTEM},
            {"role": "user", "content": user_str},
        ]

    raise ValueError(f"Unsupported dataset: {dataset}")
