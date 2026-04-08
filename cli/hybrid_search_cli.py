import argparse

from hybrid_search.hybrd_search import HybridSearch
from load_files import load_movies
from math_utils import normalize


def main() -> None:
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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize(args.numbers)
            for n in normalized:
                print(f"* {n:.4f}")
        case "weighted-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results):
                print(f"{i + 1}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.2f}")
                print(
                    f"  BM25: {result['bm25_score']:.2f}, Semantic: {result['semantic_score']:.2f}"
                )
                print(f"  {result['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
