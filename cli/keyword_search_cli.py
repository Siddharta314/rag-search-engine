#!/usr/bin/env python3

import argparse
from search_command import search
from inverted_index import InvertedIndex



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    _ = subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

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
        case "tf":
            try:
                index.load()
                tf = index.get_tf(args.doc_id, args.term)
                print(f"Term frequency for document {args.doc_id} and term '{args.term}' = {tf}")
            except FileNotFoundError:
                print("Index or docmap file not found. Please run 'python keyword_search_cli.py build' first.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
