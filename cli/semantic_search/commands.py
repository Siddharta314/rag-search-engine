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
    return parser
