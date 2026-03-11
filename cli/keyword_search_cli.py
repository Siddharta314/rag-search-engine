#!/usr/bin/env python3

import argparse
from load_movies import load_movies
from text_processing import simple_clean



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
            for movie in movies:
                if simple_clean(args.query) in simple_clean(movie["title"]):
                    print(movie["id"], movie["title"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
