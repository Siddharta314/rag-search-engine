#!/usr/bin/env python3
from semantic_search.commands import create_parser
from semantic_search.semantic_search import verify_model


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
