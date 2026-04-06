import argparse
from math_utils import normalize

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_subparser = subparsers.add_parser("normalize", help = "Normalize a sequence of numbers")
    normalize_subparser.add_argument("numbers", nargs="+", type=float, help="Numbers to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize(args.numbers)
            for n in normalized:
                print(f"* {n:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()