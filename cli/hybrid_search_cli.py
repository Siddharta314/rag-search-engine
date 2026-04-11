from hybrid_search.commands import create_parser
from hybrid_search.hybrd_search import HybridSearch
from hybrid_search.llm_utils import (
    expand_query,
    rerank_all_documents,
    rerank_all_documents_batch,
    rewrite_query,
    spell_correct,
)
from load_files import load_movies
from math_utils import normalize


def main() -> None:
    parser = create_parser()
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
            elif args.enhance == "rewrite":
                enhanced_query = rewrite_query(args.query)
                if enhanced_query.strip() == "":
                    enhanced_query = args.query
            elif args.enhance == "expand":
                enhanced_query = expand_query(args.query)
                if enhanced_query.strip() == "":
                    enhanced_query = args.query
            if args.enhance:
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
            limit = args.limit
            if args.rerank_method == "individual" or args.rerank_method == "batch":
                limit = limit * 5
            results = hs.rrf_search(enhanced_query, args.k, limit)

            if args.rerank_method == "individual":
                new_results = rerank_all_documents(enhanced_query, results)
                print(f"Re-ranking top {args.limit} results using individual method...")
                print(
                    f"Reciprocal Rank Fusion Results for '{enhanced_query}' (k={args.k}):"
                )
                for i, result in enumerate(new_results[: args.limit]):
                    print(f"{i + 1}. {result['title']}")
                    print(f"  Re-rank Score: {result['re_rank_score']:.3f}/10")
                    print(f"  RRF Score: {result['rrf_score']:.3f}")
                    print(
                        f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                    )
                    print(f"  {result['description'][:100]}...")
                    print()
                return None
            elif args.rerank_method == "batch":
                new_results = rerank_all_documents_batch(enhanced_query, results)
                print(f"Re-ranking top {args.limit} results using batch method...")
                print(
                    f"Reciprocal Rank Fusion Results for '{enhanced_query}' (k={args.k}):"
                )
                for i, result in enumerate(new_results[: args.limit]):
                    print(f"{i + 1}. {result['title']}")
                    print(f"  Re-rank Rank: {result['re_rank_rank']}")
                    print(f"  RRF Score: {result['rrf_score']:.3f}")
                    print(
                        f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                    )
                    print(f"  {result['description'][:100]}...")
                    print()
                return None

            for i, result in enumerate(results):
                print(f"{i + 1}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {result['description'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
