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
