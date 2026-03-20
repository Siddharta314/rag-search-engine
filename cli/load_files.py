import json
from typing import Any


def load_movies() -> list[dict[str, Any]]:
    """
    Load movies from movies.json file.
    
    Returns:
        list[dict[str, Any]]: List of movies with id, title, and description
    """
    movies_list_dict = []

    with open("./data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for movie in data.get("movies", []):
            m = {"id": movie["id"], "title": movie["title"], "description": movie["description"]}
            movies_list_dict.append(m)
    return movies_list_dict

def load_stopwords() -> list[str]:
    """
    Load stopwords from stopwords.txt file.
    
    Returns:
        list[str]: List of stopwords
    """
    stopwords_list = []

    with open("./data/stopwords.txt", "r", encoding="utf-8") as f:
        # Read line by line, better for larger file
        # for line in f:
        #     stopwords_list.append(line.strip())

        # Loads everything on ram - faster for small files
        stopwords_list = f.read().splitlines()

    return stopwords_list
