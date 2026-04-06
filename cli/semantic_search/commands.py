import argparse
SEARCH_DEFAULT_LIMIT = 5

def create_parser():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify the model")

    embed_parser = subparsers.add_parser("embed_text", help="Embed a text")
    embed_parser.add_argument("text", help="Text to embed")

    _ = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    query_parser = subparsers.add_parser("embedquery", help="Embed a query")
    query_parser.add_argument("query", help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search for movies")
    search_parser.add_argument("query", help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document")
    chunk_parser.add_argument("text", help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunk a document")
    semantic_chunk_parser.add_argument("text", help="Text to semantic chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, nargs='?', help="Chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, nargs="?", help="Overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("search_chunked", help="Search in chunked documents by query")
    semantic_chunk_parser.add_argument("query", help="Text to search")
    semantic_chunk_parser.add_argument("--limit", type=int, default=5, nargs='?', help="Limit the number of results")

    _ = subparsers.add_parser("embed_chunks", help="Embed chunks")
    return parser
