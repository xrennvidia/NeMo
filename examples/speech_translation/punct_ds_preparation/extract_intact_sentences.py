import argparse
import re
import string
from itertools import chain
from pathlib import Path

import nltk


WC = '\\w$\u058f\u060b\u07fe\u07ff\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\ua838\ufdfc\ufe69\uff04\uffe0\uffe1' \
    '\uffe5\uffe6°' \
    + ''.join(
        [
            chr(i) for i in chain(
                *[list(r) for r in [range(0x0a2, 0x0a6), range(0x20a1, 0x20c0), range(0x11fdd, 0x11fe1)]]
            )
        ]
    )
WORD = re.compile(f"((?:(?<=[ \n\"()])[+-]|^[+-])\\d+(?:[.,/]\\d+)*[{WC}']*|[{WC}]+(?:[.,/'][{WC}]+)*)")
LETTERS = re.compile(f"[a-zA-Z]{2,}")


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
        for line in in_f:
            sentences = nltk.sent_tokenize(line)
            if len(sentences) > 2:
                for sent in sentences[1:-1]:
                    sent = sent.strip()
                    print(sent)
                    if WORD.search(sent) is not None and LETTERS.search(sent) is not None:
                        out_f.write(sent + '\n')


if __name__ == "__main__":
    main()
