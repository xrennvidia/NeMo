import argparse
import re
from pathlib import Path


SPACE_DUP = re.compile(' {2,}')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main() -> None:
    args = get_args()
    with args.input.open() as in_f, args.output.open('w') as out_f:
        in_text = in_f.read()
        segs = in_text.split('<seg id=')[1:]
        for i, seg in enumerate(segs):
            seg = seg.split('</seg>')[0]
            seg = SPACE_DUP.sub(' ', seg[seg.index('>'):].replace('\n', ' '))
            out_f.write(seg + '\n')


if __name__ == "__main__":
    main()
