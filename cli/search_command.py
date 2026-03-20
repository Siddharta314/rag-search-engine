from nltk.stem import PorterStemmer
from load_files import load_movies, load_stopwords
from text_processing import (
    preprocess,
    compare_list_tokens,
)


def search(query: str) -> None:
    """
    Search for movies based on a query string.
    
    Args:
        query (str): The search query string.
    """
    print(f"Searching for: {query}")

    movies = load_movies()
    stopwords: list[str] = load_stopwords()
    stemmer = PorterStemmer()

    for movie in movies:
        search_token: list[str] = preprocess(query, stopwords, stemmer)
        title_token: list[str] = preprocess(movie["title"], stopwords, stemmer)

        if compare_list_tokens(search_token, title_token):
            print(f"{movie["id"]:<4}: {movie["title"]}")
