import argparse

from hybrid_search.hybrd_search import HybridSearch
from hybrid_search.llm_utils import rag_explanation, rag_summarize
from load_files import load_movies


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize a document")

    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limits the number of documents to summarize",
    )

    documents = load_movies()
    hs = HybridSearch(documents)
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = hs.rrf_search(query, 5)
            docs = "\n".join(
                [
                    f"Title: {doc.get('title', '')}: {doc.get('description', '')}"
                    for doc in results
                ]
            )
            print("Search Results:")
            for r in results:
                print(f"- {r['title']}")
            rag_response = rag_explanation(query, docs)
            print("RAG Response:")
            print(rag_response)
        case "summarize":
            query = args.query
            limit = args.limit
            results = hs.rrf_search(query, limit)
            docs = "\n".join(
                [
                    f"Title: {doc.get('title', '')}: {doc.get('description', '')}"
                    for doc in results
                ]
            )
            print("Search Results:")
            for r in results:
                print(f"- {r['title']}")
            summary = rag_summarize(query, docs)
            print("LLM Summary:")
            print(summary)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
