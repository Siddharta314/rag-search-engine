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
