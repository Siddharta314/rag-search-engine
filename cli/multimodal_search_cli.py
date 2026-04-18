import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate an embedding for an image and print its shape",
    )
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")
    search_parser = subparsers.add_parser(
        "image_search", help="Search for movies based on an image input"
    )
    search_parser.add_argument(
        "image_path", type=str, help="Path to the image to search with"
    )
    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)
    elif args.command == "image_search":
        image_search_command(args.image_path)


if __name__ == "__main__":
    main()
