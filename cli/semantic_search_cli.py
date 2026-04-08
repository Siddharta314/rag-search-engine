#!/usr/bin/env python3
from load_files import load_movies
from semantic_search.chunked_semantic_search import (
    ChunkedSemanticSearch,
    semantic_chunk,
)
from semantic_search.commands import create_parser
from semantic_search.semantic_search import (
    chunk,
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


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
        case "search_chunked":
            css = ChunkedSemanticSearch()
            movies = load_movies()
            css.load_or_create_chunk_embeddings(movies)
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results):
                print(f"\n{i + 1}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
