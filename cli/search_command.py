from nltk.stem import PorterStemmer
from load_files import load_stopwords
from text_processing import preprocess


def search(query: str, index) -> None:
    """
    Search for movies based on a query string.
    
    Args:
        query (str): The search query string.
    """
    print(f"Searching for: {query}")

    # movies = load_movies()
    stopwords: list[str] = load_stopwords()
    stemmer = PorterStemmer()
    search_token: list[str] = preprocess(query, stopwords, stemmer)
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

    ## Without the index
    # for movie in movies:
    #     title_token: list[str] = preprocess(movie["title"], stopwords, stemmer)

    #     if compare_list_tokens(search_token, title_token):
    #         print(f"{movie["id"]:<4}: {movie["title"]}")
