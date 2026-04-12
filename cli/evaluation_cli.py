import argparse

from hybrid_search.hybrd_search import HybridSearch
from load_files import load_golden_dataset, load_movies


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    documents = load_movies()
    hs = HybridSearch(documents)
    golden_dataset = load_golden_dataset()
    print(f"k={limit}")
    for case in golden_dataset:
        query = case["query"]
        results = hs.rrf_search(query, limit)
        title_results = {result["title"] for result in results}
        test_relevants = case["relevant_docs"]
        count = 0
        for relevant_doc in test_relevants:
            if relevant_doc in title_results:
                count += 1
        precision = count / limit
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(result['title'] for result in results)}")
        print(f"  - Relevant: {', '.join(test_relevants)}")
        print()


if __name__ == "__main__":
    main()
