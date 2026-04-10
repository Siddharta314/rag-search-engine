import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_subparser = subparsers.add_parser(
        "normalize", help="Normalize a sequence of numbers"
    )
    normalize_subparser.add_argument(
        "numbers", nargs="+", type=float, help="Numbers to normalize"
    )

    weighted_search = subparsers.add_parser(
        "weighted-search", help="Perform a weighted search"
    )
    weighted_search.add_argument("query", type=str, help="Query to search for")
    weighted_search.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value for the search"
    )
    weighted_search.add_argument(
        "--limit", type=int, default=5, help="Limit the number of results"
    )

    rrf_search = subparsers.add_parser("rrf-search", help="Perform a RRF search")
    rrf_search.add_argument("query", type=str, help="Query to search for")
    rrf_search.add_argument("-k", type=int, default=60, help="K value for the search")
    rrf_search.add_argument(
        "--limit", type=int, default=5, help="Limit the number of results"
    )
    rrf_search.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search.add_argument("--rerank-method", type=str, choices=["individual"])
    return parser
