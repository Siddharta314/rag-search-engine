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

    if args.command == "build":
        index.build()
        index.save()
        return
    try:
        index.load()
    except FileNotFoundError:
        print("File not found. Please run 'python keyword_search_cli.py build' first.")
        return

    match args.command:
        case "search":
            search(args.query, index)
        case "tf":
            tf = index.get_tf(args.doc_id, args.term)
            print(f"Term frequency for document {args.doc_id} and term '{args.term}' = {tf}")
        case "idf":
            idf = index.get_idf(args.term)
            print(f"Inverse document frequency for term '{args.term}' = {idf:.2f}")
        case "tfidf":
            tf = index.get_tf(args.doc_id, args.term)
            idf = index.get_idf(args.term)
            print(f"TF-IDF for document {args.doc_id} and term '{args.term}' = {tf * idf:.2f}")
        case "bm25idf":
            bm25_idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25_tf = index.get_bm25_tf(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document {args.doc_id}: {bm25_tf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
