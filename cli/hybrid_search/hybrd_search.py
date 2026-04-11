import os
from typing import TypedDict

from keyword_search.inverted_index import InvertedIndex
from math_utils import normalize
from semantic_search.chunked_semantic_search import ChunkedSemanticSearch


class WeighetedSearchResult(TypedDict):
    id: int
    title: str
    bm25_score: float
    semantic_score: float
    hybrid_score: float
    description: str


class RRFResult(TypedDict):
    id: int
    title: str
    rrf_score: float
    bm25_rank: int
    semantic_rank: int
    description: str
    re_rank_rank: int | None
    cross_encoder_score: float | None


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query, alpha: float, limit: int = 5
    ) -> list[WeighetedSearchResult]:
        bm25 = self._bm25_search(query, limit * 500)
        semantic_score = self.semantic_search.search_chunks(query, limit * 500)
        bm25_norm = normalize([b["score"] for b in bm25])
        semantic_score_norm = normalize([b["score"] for b in semantic_score])
        for i, b in enumerate(bm25):
            b["score"] = bm25_norm[i]

        for i, s in enumerate(semantic_score):
            s["score"] = semantic_score_norm[i]
        results: list[WeighetedSearchResult] = []
        combined = {}
        for b in bm25:
            combined[b["id"]] = {
                "bm25_score": b["score"],
                "semantic_score": 0.0,
                "title": b["title"],
            }

        for s in semantic_score:
            if s["id"] not in combined:
                combined[s["id"]] = {"bm25_score": 0.0}
            combined[s["id"]]["semantic_score"] = s["score"]
            combined[s["id"]]["description"] = s["document"]

        for _, data in combined.items():
            data["hybrid_score"] = hybrid_score(
                data["bm25_score"], data["semantic_score"], alpha
            )
            results.append(data)

        return sorted(results, key=lambda x: x["hybrid_score"], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10, enhance=False) -> list[RRFResult]:
        bm25 = self._bm25_search(query, limit * 500)
        semantic_score = self.semantic_search.search_chunks(query, limit * 500)
        combined = {}
        results: list[RRFResult] = []
        docs_to_map = {doc["id"]: doc for doc in self.documents}
        for i, b in enumerate(bm25, start=1):
            combined[b["id"]] = {
                "id": b["id"],
                "bm25_rank": i,
                "semantic_rank": None,
                "rrf_score": rrf_score(i, k),
                "title": b["title"],
                "description": docs_to_map[b["id"]]["description"],
            }

        for i, s in enumerate(semantic_score, start=1):
            if s["id"] not in combined:
                combined[s["id"]] = {
                    "id": s["id"],
                    "bm25_rank": None,
                    "title": s["title"],
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                    "description": docs_to_map[s["id"]]["description"],
                }
            if combined[s["id"]].get("semantic_rank") is None:
                combined[s["id"]]["semantic_rank"] = i
                combined[s["id"]]["rrf_score"] = combined[s["id"]].get(
                    "rrf_score", 0.0
                ) + rrf_score(i, k)

        for _, data in combined.items():
            results.append(data)

        return sorted(results, key=lambda x: x["rrf_score"], reverse=True)[:limit]


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)
