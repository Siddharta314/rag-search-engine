from text_processing import preprocess

from .inverted_index import InvertedIndex


def search(query: str, index: InvertedIndex) -> None:
    """Search for movies based on a query string.

    Args:
        query (str): The search query string.
        index (InvertedIndex): The inverted index to search in.

    """
    print(f"Searching for: {query}")

    search_token: list[str] = preprocess(query, index.stopwords, index.stemm)
    count = 0
    limit = 5
    for token in search_token:
        if count >= limit:
            break
        docs = index.get_document(token)
        for doc_id in docs:
            movie = index.docmap[doc_id]

            print(f"{doc_id}: {movie['title']}")
            count += 1
            if count >= limit:
                return
