from load_files import load_movies, load_stopwords
from text_processing import (
    simple_clean,
    tokenize_based_word,
    compare_list_tokens,
    remove_stopwords
)
def search(query: str) -> None:
    print(f"Searching for: {query}")

    movies = load_movies()
    stopwords: list[str] = load_stopwords()

    for movie in movies:
        search_clean = remove_stopwords(simple_clean(query), stopwords)
        title_clean = remove_stopwords(simple_clean(movie["title"]), stopwords)
        if compare_list_tokens(
            tokenize_based_word(search_clean),
            tokenize_based_word(title_clean)
        ):
            print(movie["id"], movie["title"])
