#!/usr/bin/env python3
from semantic_search.commands import create_parser
from semantic_search.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
