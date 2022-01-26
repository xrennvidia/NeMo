import argparse
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_mwer", type=Path, required=True)
    parser.add_argument("--input_orig", type=Path, required=True)
    parser.add_argument("--output_mwer", type=Path, required=True)
    parser.add_argument("--output_orig", type=Path, required=True)
    args = parser.parse_args()
    args.input_mwer = args.input_mwer.expanduser()
    args.input_orig = args.input_orig.expanduser()
    args.output_mwer = args.output_mwer.expanduser()
    args.output_orig = args.output_orig.expanduser()
    return args


def main() -> None:
    args = get_args()
    with args.input_mwer.open():
        pass


if __name__ == "__main__":
    main()