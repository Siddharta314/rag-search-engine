import json

from hybrid_search.commands import create_parser, print_results
from hybrid_search.hybrd_search import HybridSearch
from hybrid_search.llm_utils import (
    evaluate_results,
    expand_query,
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
            if (
                args.rerank_method == "individual"
                or args.rerank_method == "batch"
                or args.rerank_method == "cross_encoder"
            ):
                limit = limit * 5
            results = hs.rrf_search(enhanced_query, limit, args.k)
            print_results(args, enhanced_query, results)
            if args.evaluate:
                try:
                    json_scores_llm = evaluate_results(enhanced_query, results)
                    print(json_scores_llm)
                    llm_evaluation = json.loads(json_scores_llm)
                except Exception as e:
                    print(f"Error in the llm evaluation: {e}")
                    return None
                if len(llm_evaluation) != len(results):
                    print("Error in the llm evaluation")
                    return None

                for i, (result, score) in enumerate(
                    zip(results, llm_evaluation, strict=False)
                ):
                    print(f"{i + 1}. {result['title']}: {score}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
