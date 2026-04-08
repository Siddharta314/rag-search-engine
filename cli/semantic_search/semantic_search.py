from typing import Any

import numpy as np
from load_files import load_movies
from math_utils import cosine_similarity
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict[str, Any]] | None = None
        self.document_map: dict[int, dict[str, Any]] = {}

    def generate_embedding(self, text) -> np.ndarray:
        clean_text = text.strip()
        if clean_text == "":
            raise ValueError("Text cannot be empty")
        result = self.model.encode([clean_text])
        return result[0]

    def build_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        doc_string_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_string_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_string_list, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        try:
            loaded_embeddings = np.load("cache/movie_embeddings.npy")
            if len(loaded_embeddings) == len(documents):
                self.embeddings = loaded_embeddings
                return self.embeddings
        except (FileNotFoundError, ValueError):
            return self.build_embeddings(documents)

    def search(self, query, limit) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        pairs = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            pairs.append((similarity, self.documents[i]))
        pairs.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": pairs[i][0],
                "title": pairs[i][1]["title"],
                "description": pairs[i][1]["description"],
            }
            for i in range(len(pairs[:limit]))
        ]


def verify_model() -> bool:
    try:
        semantic_search = SemanticSearch()
        model = semantic_search.model
        print(f"Model loaded: {model}")
        print(f"Max sequence length: {model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    return True


def embed_text(text):
    semantic_search = SemanticSearch()
    print(f"Text: {text}")
    embedding = semantic_search.generate_embedding(text)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {len(embedding)}")
    return embedding


def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    if semantic_search.embeddings is None or semantic_search.documents is None:
        print("Embeddings or documents not loaded.")
        return False
    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(
        f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions"
    )
    return True


def embed_query_text(query):
    semantic_search = SemanticSearch()
    print(f"Query: {query}")
    embedding = semantic_search.generate_embedding(query)
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding


def search(query, limit):
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    results = semantic_search.search(query, limit)
    for index, item in enumerate(results):
        print(
            f"{index + 1}. {item['title']} (score: {item['score']:.4f})\n   {item['description']}"
        )
    return results


def chunk(text: str, chunk_size: int = 200, overlap: int = 0):
    split_text = text.split()
    chunks = []
    i = 0
    while i < len(split_text):
        chunks.append(" ".join(split_text[i : i + chunk_size]))
        if i + chunk_size >= len(split_text):
            break
        i += chunk_size - overlap
    print(f"Chunking {len(text)} characters")
    for i, c in enumerate(chunks):
        print(f"{i + 1}. {c}")
    return chunks
