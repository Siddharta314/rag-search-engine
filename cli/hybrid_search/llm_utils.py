import json
import os
import time

from dotenv import load_dotenv
from google import genai
from hybrid_search.hybrd_search import RRFResult

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)


def spell_correct(query: str) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
""",
    )
    return (response.text or "").strip()


def rewrite_query(query: str) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
""",
    )
    return (response.text or "").strip()


def expand_query(query: str) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"
- "math movie" -> "intelligence iq, math magic"

User query: "{query}"
""",
    )
    return (response.text or "").strip()


def rerank_individual(query: str, doc: RRFResult) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:""",
    )
    return (response.text or "").strip()


def rerank_in_batch(query: str, docs: list[RRFResult]) -> str:
    doc_list_str = "\n".join(
        [
            f"Movie ID: {doc.get('id', '')} - {doc.get('title', '')} - {doc.get('description', '')}"
            for doc in docs
        ]
    )
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:""",
    )
    return (response.text or "").strip()


def rerank_all_documents(query: str, documents: list[RRFResult]) -> list:
    results = []
    for doc in documents:
        rerank_score = rerank_individual(query, doc)
        try:
            rerank_score = int(rerank_score)
        except ValueError:
            rerank_score = 0
        results.append({**doc, "re_rank_score": rerank_score})
        time.sleep(3)
    return sorted(results, key=lambda x: x["re_rank_score"], reverse=True)


def rerank_all_documents_batch(query: str, documents: list) -> list:
    results_ids = json.loads(rerank_in_batch(query, documents)) or []
    results = []
    for i, result in enumerate(results_ids):
        for doc in documents:
            if doc["id"] == result:
                results.append({**doc, "re_rank_rank": i + 1})
    return results


def rag_explanation(query: str, docs: str) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:""",
    )
    return (response.text or "").strip()


def rag_summarize(query: str, docs: str) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:""",
    )
    return (response.text or "").strip()


def rag_citations(query: str, docs: str) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:""",
    )
    return (response.text or "").strip()


def rag_answering_questions(query: str, docs: str) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:""",
    )
    return (response.text or "").strip()
