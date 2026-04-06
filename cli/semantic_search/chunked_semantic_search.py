import re
import json
import numpy as np
from .semantic_search import SemanticSearch
from math_utils import cosine_similarity 

SCORE_PRECISION = 4
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        chunks = []
        self.chunk_embeddings = []
        self.chunk_metadata = []
        self.documents = documents
        for idx, doc in enumerate(self.documents):
            movie_id = doc["id"] 
            if doc["description"] is None:
                continue
            self.document_map[movie_id] = doc
            chunks_doc = semantic_chunk(doc["description"], 4, 1)
            chunks.extend(chunks_doc)
            for i in range(len(chunks_doc)):
                self.chunk_metadata.append({
                    "movie_idx": movie_id,
                    "chunk_idx": i,
                    "total_chunks": len(chunks_doc)
                })
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        try:
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings if len(self.chunk_embeddings) == len(self.chunk_metadata) else self.build_chunk_embeddings(documents)
        except (FileNotFoundError, json.JSONDecodeError):
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedded = self.generate_embedding(query)
        chunk_scores = []
        for idx, chunk in enumerate(self.chunk_embeddings):
            chunk_scores.append({
                "chunk_idx": idx,
                "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                "score": cosine_similarity(query_embedded, chunk)
            })
        movies_index_score = {}
        for chunk in chunk_scores:
            movie_index = chunk["movie_idx"]
            current_score = chunk["score"]
            if movie_index not in movies_index_score or movies_index_score[movie_index] < current_score:
                movies_index_score[movie_index] = current_score
        sorted_movies = sorted(movies_index_score.items(), key=lambda x: x[1], reverse=True)
        results = []
        for movie_index, score in sorted_movies[:limit]:
            movie_data = self.document_map[movie_index]
            results.append({
            "id": movie_index,
            "title": movie_data["title"],
            "document": movie_data["description"][:100],
            "score": round(score, SCORE_PRECISION), 
            "metadata": movie_data.get("metadata", {})
        })
        return results

def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0):
    text = text.strip()
    
    # 2. Si después de limpiar no queda nada, salimos sin imprimir "1. "
    if not text:
        print("Semantically chunking 0 characters")
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    i = 0
    while i < len(sentences):
        chunks.append(" ".join(sentences[i:i + max_chunk_size]))
        if i + max_chunk_size >= len(sentences):
            break
        i += max_chunk_size - overlap

    print(f"Semantically chunking {len(text)} characters")
    for i, c in enumerate(chunks):
        print(f"{i + 1}. {c}")
    return chunks
