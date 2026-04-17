import argparse
import mimetypes
import os

from google.genai import types
from hybrid_search.llm_utils import MODEL, client


def main():
    parser = argparse.ArgumentParser(description="Multimodal query rewriting")
    parser.add_argument("--image", type=str, required=True, help="Path of the image")
    parser.add_argument(
        "--query", type=str, required=True, help="A text query to rewrite"
    )

    args = parser.parse_args()
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    with open(args.image, "rb") as f:
        img_data: bytes = f.read()
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    parts: list[types.Part] = [
        types.Part.from_text(text=system_prompt),
        types.Part.from_bytes(data=img_data, mime_type=mime),
        types.Part.from_text(text=args.query.strip()),
    ]

    config = types.GenerateContentConfig(temperature=0.5)

    response = client.models.generate_content(
        model=MODEL,
        contents=parts,  # En la versión actual de la API se suele usar 'contents'
        config=config,
    )

    if response.text:
        print(f"\nRewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
