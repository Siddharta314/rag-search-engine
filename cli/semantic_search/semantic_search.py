import numpy as np
from sentence_transformers import SentenceTransformer
from load_files import load_movies

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        clean_text = text.strip()
        if clean_text == "":
            raise ValueError("Text cannot be empty")
        result = self.model.encode([clean_text])
        return result[0]

    def build_embeddings(self, documents):
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
            self.embeddings = np.load("cache/movie_embeddings.npy")
            return self.embeddings if len(self.embeddings) == len(documents) else self.build_embeddings(documents)
        except FileNotFoundError:
            return self.build_embeddings(documents)


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
    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")
    return True


def embed_query_text(query):
    semantic_search = SemanticSearch()
    print(f"Query: {query}")
    embedding = semantic_search.generate_embedding(query)
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding