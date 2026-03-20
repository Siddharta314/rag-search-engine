#!/usr/bin/env python3

import argparse
from search_command import search
from inverted_index import InvertedIndex



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = subparsers.add_parser("build", help="Build the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    index = InvertedIndex()

    match args.command:
        case "search":
            try:
                index.load()
                search(args.query, index)
            except FileNotFoundError:
                print("Index or docmap file not found. Please run 'python keyword_search_cli.py build' first.")
        case "build":
            index.build()
            index.save()
            docs = index.get_document("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
