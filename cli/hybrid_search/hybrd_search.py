import os
from math_utils import normalize
from keyword_search.inverted_index import InvertedIndex
from semantic_search.chunked_semantic_search import ChunkedSemanticSearch

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

    def weighted_search(self, query, alpha: float, limit: int=5)-> list[dict]:
        bm25 = self._bm25_search(query, limit*500)
        semantic_score = self.semantic_search.search_chunks(query, limit*500)
        bm25_norm = normalize([b["score"] for b in bm25])
        semantic_score_norm = normalize([b["score"] for b in semantic_score])
        doc_map = {doc["id"]: doc for doc in self.documents}
        for i, b in enumerate(bm25):
            b["normalized_score"] = bm25_norm[i]

        for i, s in enumerate(semantic_score):
            s["normalized_score"] = semantic_score_norm[i]

        result = []
        combined = {}
        for b in bm25:
            description = doc_map[b["id"]]["description"]
            combined[b["id"]] = { "bm25_score": b["normalized_score"], "semantic_score": 0.0, "title": b["title"], "description": description }

        for s in semantic_score:
            if s["id"] not in combined:
                combined[s["id"]] = { "bm25_score": 0.0, "text": s["text"] }
            combined[s["id"]]["semantic_score"] = s["normalized_score"]

        for _, data in combined.items():
            data["hybrid_score"] = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
            result.append(data)

        return sorted(result, key=lambda x: x["hybrid_score"], reverse = True)[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")



def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score
