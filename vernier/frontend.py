"""Frozen zero-LLM-build front-end: BGE-M3 + TF-IDF RRF (Pool-80) -> BaseProvence (Top-32)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from vernier.packing import render_inline_time_grounding


@dataclass
class VerbatimIndex:
    turns: list[dict[str, Any]]
    dense_vecs: np.ndarray
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: Any


class FrozenFrontEnd:
    """Off-the-shelf frozen retrieval and reranking front-end (zero offline LLM calls)."""

    def __init__(
        self,
        bge_model_name: str = "BAAI/bge-m3",
        reranker_model_name: str = "naver/provence-reranker-debertav3-v1",
        selector_tokenizer_name: str = "Qwen/Qwen3-1.7B",
        device: str = "cuda",
        rrf_k: int = 60,
        pool_size: int = 80,
        top_m: int = 32,
    ) -> None:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.rrf_k = rrf_k
        self.pool_size = pool_size
        self.top_m = top_m

        self.bge_tok = AutoTokenizer.from_pretrained(bge_model_name)
        self.bge_model = AutoModel.from_pretrained(bge_model_name, dtype=torch.bfloat16).to(device).eval()

        self.rerank_tok = AutoTokenizer.from_pretrained(reranker_model_name)
        self.rerank_model = (
            AutoModelForSequenceClassification.from_pretrained(
                reranker_model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        self.sel_tok = AutoTokenizer.from_pretrained(selector_tokenizer_name)

    @torch.inference_mode()
    def _encode_bge(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = []
        for st in range(0, len(texts), batch_size):
            batch = texts[st : st + batch_size]
            enc = self.bge_tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            out = self.bge_model(**enc).last_hidden_state[:, 0]
            out = torch.nn.functional.normalize(out.float(), p=2, dim=-1)
            vecs.append(out.cpu().numpy())
        return np.vstack(vecs) if vecs else np.zeros((0, 1024), dtype=np.float32)

    def build_index(self, turns: list[dict[str, Any]]) -> VerbatimIndex:
        """Index verbatim dialogue turns offline using frozen BGE-M3 + sublinear TF-IDF."""
        rendered = [
            f"[{tr.get('iso_date', '')}] {tr['speaker']}: {tr['text']}"[:1000]
            for tr in turns
        ]
        dense_vecs = self._encode_bge(rendered)
        vec_tf = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            max_features=200000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        )
        tfidf_matrix = vec_tf.fit_transform(rendered)
        return VerbatimIndex(
            turns=turns,
            dense_vecs=dense_vecs,
            tfidf_vectorizer=vec_tf,
            tfidf_matrix=tfidf_matrix,
        )

    @torch.inference_mode()
    def retrieve_top_m(self, index: VerbatimIndex, question: str) -> list[dict[str, Any]]:
        """Retrieve Pool-80 via RRF (k=60) and rerank to Top-32 via frozen BaseProvence."""
        n_t = len(index.turns)
        if n_t == 0:
            return []

        q_vec = self._encode_bge([question])[0]
        d_sc = index.dense_vecs @ q_vec
        q_tf = index.tfidf_vectorizer.transform([question])
        l_sc = np.asarray((index.tfidf_matrix @ q_tf.T).toarray()).ravel()

        k_lim = min(n_t, 200)
        d_top = np.argpartition(-d_sc, k_lim - 1)[:k_lim]
        d_top = d_top[np.argsort(-d_sc[d_top], kind="stable")]
        l_top = np.argpartition(-l_sc, k_lim - 1)[:k_lim]
        l_top = l_top[np.argsort(-l_sc[l_top], kind="stable")]

        rrf: dict[int, float] = defaultdict(float)
        for rk, idx_t in enumerate(d_top, 1):
            rrf[int(idx_t)] += 1.0 / (self.rrf_k + rk)
        for rk, idx_t in enumerate(l_top, 1):
            if l_sc[idx_t] > 0:
                rrf[int(idx_t)] += 1.0 / (self.rrf_k + rk)

        pool_ids = sorted(rrf.keys(), key=lambda i: (-rrf[i], -float(d_sc[i]), i))[: self.pool_size]
        pairs = [
            (
                question,
                f"[{index.turns[i].get('date_hdr', '')} | {index.turns[i]['speaker']}] {index.turns[i]['text']}",
            )
            for i in pool_ids
        ]

        rerank_scores: list[float] = []
        for st in range(0, len(pairs), 32):
            sub = pairs[st : st + 32]
            enc = self.rerank_tok(
                [p[0] for p in sub],
                [p[1] for p in sub],
                padding=True,
                truncation=True,
                max_length=448,
                return_tensors="pt",
            ).to(self.device)
            out = self.rerank_model(**enc)
            logits = out.logits.squeeze(-1) if hasattr(out, "logits") else out[0].squeeze(-1)
            rerank_scores.extend(logits.float().cpu().tolist())

        scored = sorted(
            zip(pool_ids, rerank_scores),
            key=lambda x: (-x[1], -rrf[x[0]]),
        )[: self.top_m]

        candidates: list[dict[str, Any]] = []
        for tp, sc in scored:
            tr = index.turns[tp]
            iso_d = str(tr.get("iso_date") or "")
            txt = render_inline_time_grounding(str(tr["text"]), iso_d)
            vr = f"{tr['speaker']}: {txt}"
            tok_len = len(self.sel_tok.encode(vr, add_special_tokens=False)) + 6
            candidates.append(
                {
                    "t_pos": int(tr.get("t_pos", tp)),
                    "source_id": str(tr.get("source_id", f"t{tp}")),
                    "sess_id": str(tr.get("sess_id", f"s{tr.get('sess_ordinal', 0)}")),
                    "sess_ordinal": int(tr.get("sess_ordinal", 0)),
                    "iso_date": iso_d,
                    "date_hdr": str(tr.get("date_hdr") or iso_d),
                    "speaker": str(tr["speaker"]),
                    "stage1_rrf": round(float(rrf[tp]), 6),
                    "rerank_score": round(float(sc), 6),
                    "tok_len": int(tok_len),
                    "verbatim_render": vr,
                }
            )
        return candidates
