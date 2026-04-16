import argparse

from hybrid_search.llm_utils import (
    rerank_all_documents,
    rerank_all_documents_batch,
)
from sentence_transformers import CrossEncoder


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
    rrf_search.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"]
    )
    rrf_search.add_argument("--evaluate", action="store_true", default=False)
    return parser


def print_results(args, enhanced_query, results):
    if args.rerank_method:
        print(
            f"Re-ranking top {args.limit} results using {args.rerank_method} method..."
        )
        print(f"Reciprocal Rank Fusion Results for '{enhanced_query}' (k={args.k}):")
        new_results = []
        if args.rerank_method == "individual":
            new_results = rerank_all_documents(enhanced_query, results)
        elif args.rerank_method == "batch":
            new_results = rerank_all_documents_batch(enhanced_query, results)
        elif args.rerank_method == "cross_encoder":
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            pairs = [
                [
                    args.query,
                    f"{doc.get('title', '')} - {doc.get('document', '')}",
                ]
                for doc in results
            ]
            scores = cross_encoder.predict(pairs)
            for i in range(len(results)):
                results[i]["cross_encoder_score"] = scores[i]
            new_results = sorted(
                results,
                key=lambda x: x.get("cross_encoder_score", 0.0) or 0.0,
                reverse=True,
            )
        for i, result in enumerate(new_results[: args.limit]):
            print(f"{i + 1}. {result['title']}")
            if args.rerank_method == "individual":
                print(f"  Re-rank Score: {result['re_rank_score']:.3f}/10")
            elif args.rerank_method == "batch":
                print(f"  Re-rank Rank: {result['re_rank_rank']}")
            elif args.rerank_method == "cross_encoder":
                print(f"  Cross Encoder Score: {result['cross_encoder_score']:.3f}")
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
