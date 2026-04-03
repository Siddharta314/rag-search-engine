#!/usr/bin/env python3
from semantic_search.commands import create_parser
from semantic_search.semantic_search import (
    verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk
)
from semantic_search.chunked_semantic_search import semantic_chunk, ChunkedSemanticSearch
from load_files import load_movies

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
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            css = ChunkedSemanticSearch()
            movies = load_movies()
            embeddings = css.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
