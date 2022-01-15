import argparse
from pathlib import Path

from nemo_text_processing.text_normalization.normalize import Normalizer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_file = args.output_file.expanduser()
    return args


def main() -> None:
    args = get_args()
    normalizer = Normalizer(input_case='cased', lang='en')
    with args.input_file.open() as f:
        lines = f.readlines()
    with args.output_file.open('w') as f:
        for line in normalizer.normalize_list(lines):
            f.write(line + ('' if line and line[-1] == '\n' else '\n'))


if __name__ == "__main__":
    main()
