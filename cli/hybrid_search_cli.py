import argparse

from hybrid_search.hybrd_search import HybridSearch
from hybrid_search.llm_utils import rewrite_query, spell_correct
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

    rrf_search = subparsers.add_parser("rrf-search", help="Perform a RRF search")
    rrf_search.add_argument("query", type=str, help="Query to search for")
    rrf_search.add_argument("-k", type=int, default=60, help="K value for the search")
    rrf_search.add_argument(
        "--limit", type=int, default=5, help="Limit the number of results"
    )
    rrf_search.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
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
                print(f"  {result['description']}...")
        case "rrf-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            enhanced_query = args.query
            if args.enhance == "spell":
                enhanced_query = spell_correct(args.query)
                if enhanced_query.strip() == "":
                    enhanced_query = args.query
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
            elif args.enhance == "rewrite":
                enhanced_query = rewrite_query(args.query)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
                if enhanced_query.strip() == "":
                    enhanced_query = args.query
            results = hs.rrf_search(enhanced_query, args.k, args.limit)

            for i, result in enumerate(results):
                print(f"{i + 1}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.4f}")
                print(
                    f"  BM25: {result['bm25_rank']}, Semantic: {result['semantic_rank']}"
                )
                print(f"  {result['description']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
