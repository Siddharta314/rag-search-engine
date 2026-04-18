from typing import Any, TypedDict

import numpy as np
from load_files import load_movies
from PIL import Image
from sentence_transformers import SentenceTransformer


class SearchResult(TypedDict):
    id: Any
    title: str
    description: str
    score: float


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}\n" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str) -> list[SearchResult]:
        image_embedding = self.embed_image(image_path)

        results: list[SearchResult] = []
        dot_product = np.dot(self.text_embeddings, image_embedding)

        norms = np.linalg.norm(self.text_embeddings, axis=1) * np.linalg.norm(
            image_embedding
        )
        similarities = dot_product / norms
        for i, score in enumerate(similarities):
            doc = self.documents[i]
            search_result: SearchResult = {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
                "score": float(score),
            }
            results.append(search_result)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:5]


def verify_image_embedding(image_path):
    docs = []
    ms = MultimodalSearch(docs)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path) -> list[SearchResult]:
    movies = load_movies()
    ms = MultimodalSearch(movies)
    results = ms.search_with_image(image_path)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['title']} (similarity: {result['score']:.3f})")
        print(f"{result['description'][:100]}...")

    return results
