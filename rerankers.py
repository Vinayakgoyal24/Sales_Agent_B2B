from __future__ import annotations
import time
from typing import List, Dict
from langchain_core.documents import Document
import cohere
from typing import List
import os
from langchain.schema import Document

TOP_K   = 15        # docs taken from vector store
FINAL_K = 5         # docs kept after rerank
# rerankers.py
TOP_K = 10

from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.indices.postprocessor.colbert_rerank import ColbertRerank

import time, textwrap
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.keyword_overlap_postprocessor import KeywordOverlapPostprocessor

class ColBERTReranker(BaseReranker):
    """
    ColBERT-v2 late-interaction scorer using Llama-Index post-processor.

    * Checkpoint: colbert-ir/colbertv2.0 (110 MB)
    * top_n = FINAL_K
    * Prints every passage + rerank score for debugging.
    """

    def _init_(self,
                 ckpt: str = "colbert-ir/colbertv2.0",
                 top_n: int = FINAL_K):
        super()._init_("ColBERT-v2")
        self.top_n = top_n
        self.rerank = ColbertRerank(
            top_n=top_n,
            model=ckpt,
            tokenizer=ckpt,
            keep_retrieval_score=True,
        )

    # util to shorten lines in prints
    def _short(self, txt: str, n: int = 120):
        return textwrap.shorten(txt.replace("\n", " "), width=n, placeholder="…")

    def _call_(self, query: str, docs: List[Document]) -> List[Document]:
        print(f"\n[ColBERT] Query: {query}")
        print(f"[ColBERT] Passages in: {len(docs)}")

        # Step 1 ─ build NodeWithScore list (score dummy 0.0)
        node_list: list[NodeWithScore] = []
        for idx, d in enumerate(docs):
            tn = TextNode(id_=str(idx), text=d.page_content, metadata=d.metadata or {})
            node_list.append(NodeWithScore(node=tn, score=0.0))
            print(f"  » id={idx:<3} {self._short(d.page_content)}")

        # Step 2 ─ call ColbertRerank
        t0 = time.perf_counter()
        ranked_nodes: list[NodeWithScore] = self.rerank.postprocess_nodes(
            node_list, query_str=query
        )
        t1 = time.perf_counter()

        # Step 3 ─ top_n + diagnostics
        ranked_nodes = ranked_nodes[: self.top_n]
        print(f"[ColBERT] Top {self.top_n} after rerank ({round((t1-t0)*1000,1)} ms):")
        picked_docs, scores = [], []
        for rank, nw in enumerate(ranked_nodes, 1):
            orig_idx = int(nw.node.id_)
            picked_docs.append(docs[orig_idx])
            scores.append(nw.score)
            print(f"  #{rank} id={orig_idx:<3} score={nw.score:.4f} {self._short(docs[orig_idx].page_content)}")

        self._set_metrics(t0, t1, docs, picked_docs, scores)
        return picked_docs


class CohereReranker:
    def __init__(self, api_key: str, top_n: int = 5):
        self.client = cohere.Client(api_key)
        self.top_n = top_n
        self._last_scores = []

    def __call__(self, query: str, docs: List[Document]) -> List[Document]:
        passages = [doc.page_content for doc in docs]
        response = self.client.rerank(query=query, documents=passages, top_n=self.top_n, model="rerank-english-v2.0")
        self._last_scores = [r["relevance_score"] for r in response.results]
        return [docs[r["index"]] for r in response.results]

    def metrics(self):
        return {"avg_relevance_score": sum(self._last_scores) / len(self._last_scores) if self._last_scores else 0}

# ───────────────────────── FlashRank (0.2.10) imports ───────────────────────
try:
    from flashrank import Ranker, RerankRequest
except ImportError as e:
    raise RuntimeError("🔧  pip install flashrank==0.2.10") from e

# ───────────────────────── Base with simple metrics ──────────────────────────
class BaseReranker:
    def __init__(self, name: str):
        self.name = name
        self._metrics: Dict = {}

    def metrics(self) -> Dict:
        return self._metrics

    def _set_metrics(self, t0, t1, docs_in, ranked, scores):
        self._metrics = dict(
            model=self.name,
            latency_ms=round((t1 - t0) * 1000, 1),
            k_in=len(docs_in),
            k_out=len(ranked),
            scores=scores,
        )

from sentence_transformers import CrossEncoder
import numpy as np, torch

class BGEReranker(BaseReranker):
    """
    Uses Sentence-Transformers CrossEncoder so we don’t depend on FlashRank zips.
    """

    def __init__(self):
        super().__init__("BGE-reranker-base")
        # downloads once to ~/.cache/huggingface/…
        self.ce = CrossEncoder("BAAI/bge-reranker-base", device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, query: str, docs: List[Document]) -> List[Document]:
        pairs = [[query, d.page_content] for d in docs]

        t0 = time.perf_counter()
        scores = self.ce.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        t1 = time.perf_counter()

        order = np.argsort(-scores)[:FINAL_K]          # top-K, highest first
        picked = [docs[i] for i in order]
        sel_scores = [float(scores[i]) for i in order]

        self._set_metrics(t0, t1, docs, picked, sel_scores)
        return picked

# ───────────────────────── Rank-T5 implementation ────────────────────────────
class RankT5Reranker(BaseReranker):
    """
    Uses FlashRank 0.2.10 signature:

        req = RerankRequest(query, passages)
        ranked = ranker.rerank(req)

    Where each passage is { "id": int, "text": str, "meta": dict|null }
    """
    def __init__(self):
        super().__init__("Rank-T5-Flan")
        # 110 MB checkpoint will download on first run to ~/.cache/torch
        self.rank = Ranker(model_name="rank-T5-flan")

    def __call__(self, query: str, docs: List[Document]) -> List[Document]:
        # 1️⃣ prepare passages list
        passages = [
            {"id": idx, "text": d.page_content, "meta": d.metadata or {}}
            for idx, d in enumerate(docs)
        ]

        # 2️⃣ build request → call reranker
        req = RerankRequest(query=query, passages=passages)

        t0 = time.perf_counter()
        ranked = self.rank.rerank(req)[:FINAL_K]         # ← single positional arg
        t1 = time.perf_counter()

        # 3️⃣ map back to original Document objects
        ids    = [item["id"] for item in ranked]
        scores = [item["score"] for item in ranked]
        picked = [docs[i] for i in ids]

        self._set_metrics(t0, t1, docs, picked, scores)
        return picked

# ───── ColBERT-v2 (~110 MB) ────────────────────────────────────────────────
class ColBERTReranker(BaseReranker):
    def __init__(self):
        super().__init__("ColBERT-v2")
        self.rank = Ranker(model_name="colbert-ir/colbertv2.0")

    def __call__(self, query: str, docs: List[Document]) -> List[Document]:
        # FlashRank expects list[(text, meta)]
        passages = [(d.page_content, {}) for d in docs]
        req = RerankRequest(query=query, passages=passages)  # same object
        t0 = time.perf_counter()
        ranked = self.rank.rerank(req)[:FINAL_K]
        t1 = time.perf_counter()

        ids     = [r["id"] for r in ranked]          # id field is 1-based for this model
        scores  = [r["score"] for r in ranked]
        picked  = [docs[i-1] for i in ids]

        self._set_metrics(t0, t1, docs, picked, scores)
        return picked

# ───── Rank-Zephyr 7B (4-bit, GPU recommended) ─────────────────────────────
class RankZephyrReranker(BaseReranker):
    def __init__(self):
        super().__init__("Rank-Zephyr-7B")
        self.rank = Ranker(model_name="rank_zephyr_7b_v1_full", max_length=1024)

    def __call__(self, query: str, docs: List[Document]) -> List[Document]:
        # ids start at 1 to satisfy FlashRank
        passages = [
            {"id": i + 1, "text": d.page_content, "meta": d.metadata or {}}
            for i, d in enumerate(docs)
        ]
        req = RerankRequest(query=query, passages=passages)

        t0 = time.perf_counter()
        raw = self.rank.rerank(req)[:FINAL_K]
        t1 = time.perf_counter()

        ids, scores = [], []
        for item in raw:
            if isinstance(item, dict):            # normal case for future versions
                ids.append(int(item["id"]) - 1)
                scores.append(item.get("score"))
            else:                                 # string "[9]" or int 9
                idx = int(str(item).strip("[]")) - 1
                ids.append(idx)
                scores.append(None)

        picked = [docs[i] for i in ids]
        self._set_metrics(t0, t1, docs, picked, scores)
        return picked

# ───────────────────────── MiniLM-L-12-v2 (34 MB) ────────────────────────────
class MiniLMReranker(BaseReranker):
    """
    microsoft/ms-marco-MiniLM-L-12-v2  –  tiny & fast cross-encoder.
    """

    def __init__(self):
        super().__init__("MiniLM-L-12-v2")
        self.rank = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def __call__(self, query: str, docs: List[Document]) -> List[Document]:
        # FlashRank expects list[dict(id,text,meta)], 1-indexed ids
        passages = [
            {"id": i , "text": d.page_content, "meta": d.metadata or {}}
            for i, d in enumerate(docs)
        ]
        req = RerankRequest(query=query, passages=passages)

        t0 = time.perf_counter()
        ranked = self.rank.rerank(req)[:FINAL_K]     # ← no crash now
        t1 = time.perf_counter()

        ids    = [item["id"] for item in ranked]   # back to 0-based
        scores = [item["score"]   for item in ranked]
        picked = [docs[i]         for i in ids]

        self._set_metrics(t0, t1, docs, picked, scores)
        return picked
def get_reranker(name: str) -> BaseReranker:
    name = name.lower()
    if name.lower() == "cohere":
        return CohereReranker(api_key=os.getenv("COHERE_API_KEY"), top_n=5)
    if name == "rankt5":     return RankT5Reranker()
    if name == "colbert":    return ColBERTReranker()
    if name == "rankzephyr": return RankZephyrReranker()
    if name == "minilm":     return MiniLMReranker()
    if name == "bge":        return BGEReranker() 
    if name == "colbert":    return ColBERTReranker()     
    raise ValueError(f"unknown reranker '{name}'")