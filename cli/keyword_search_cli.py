#!/usr/bin/env python3

import argparse
from load_files import load_movies, load_stopwords
from text_processing import (
    simple_clean, 
    tokenize_based_word, 
    compare_list_tokens,
    remove_stopwords
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            movies = load_movies()
            stopwords: list[str] = load_stopwords()

            for movie in movies:
                search_clean = remove_stopwords(simple_clean(args.query), stopwords)
                title_clean = remove_stopwords(simple_clean(movie["title"]), stopwords)
                if compare_list_tokens(
                    tokenize_based_word(search_clean),
                    tokenize_based_word(title_clean)
                ):
                    print(movie["id"], movie["title"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
