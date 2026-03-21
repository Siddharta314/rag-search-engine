#!/usr/bin/env python3
"""
CLI for keyword search engine
"""
from commands import create_parser
from search_command import search
from inverted_index import InvertedIndex


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    index = InvertedIndex()

    match args.command:
        case "search":
            try:
                index.load()
                search(args.query, index)
            except FileNotFoundError:
                print("File not found. Please run 'python keyword_search_cli.py build' first.")
        case "build":
            index.build()
            index.save()
        case "tf":
            try:
                index.load()
                tf = index.get_tf(args.doc_id, args.term)
                print(f"Term frequency for document {args.doc_id} and term '{args.term}' = {tf}")
            except FileNotFoundError:
                print("File not found. Please run 'python keyword_search_cli.py build' first.")
        case "idf":
            try:
                index.load()
                idf = index.get_idf(args.term)
                print(f"Inverse document frequency for term '{args.term}' = {idf:.2f}")
            except FileNotFoundError:
                print("File not found. Please run 'python keyword_search_cli.py build' first.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
