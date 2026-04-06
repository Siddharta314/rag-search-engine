import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def normalize(numbers: list[float]) -> list[float]:
    if not numbers:
        return []
    min_val = min(numbers)
    max_val = max(numbers)
    if min_val == max_val:
        return [1.0] * len(numbers)
    return [(x - min_val) / (max_val - min_val) for x in numbers]