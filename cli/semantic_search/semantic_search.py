from sentence_transformers import SentenceTransformer

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        clean_text = text.strip()
        if clean_text == "":
            raise ValueError("Text cannot be empty")
        result = self.model.encode([clean_text])
        return result[0]

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