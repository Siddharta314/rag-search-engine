#!/usr/bin/env python3

import argparse
from search_command import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = subparsers.add_parser("build", help="Build the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
        case "build":
            from inverted_index import InvertedIndex
            index = InvertedIndex()
            index.build()
            index.save()
            docs = index.get_document("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
