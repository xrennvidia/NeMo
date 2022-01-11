import sys
sys.path = ["/home/apeganov/NeMo"] + sys.path

import argparse
import json
import os
import re
from pathlib import Path

from nemo.collections.nlp.models import PunctuationCapitalizationModel


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
SPACE_DEDUP = re.compile(r' +')
LONG_NUMBER = re.compile(r"[1-9][0-9]{3,}")
PUNCTUATION = re.compile("[.,?]")
DECIMAL = re.compile(f"[0-9]+{PUNCTUATION.pattern}? point({PUNCTUATION.pattern}? [0-9])+", flags=re.I)
LEFT_PUNCTUATION_STRIP_PATTERN = re.compile('^[^a-zA-Z]+')
RIGHT_PUNCTUATION_STRIP_PATTERN = re.compile('[^a-zA-Z]$')

MAX_NUM_SUBTOKENS_IN_INPUT = 8192


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-to-align-with", "-a", required=True, type=Path)
    parser.add_argument(
        "--model", "-m", default="punctuation_en_bert", help="Path to .nemo file or name of pretrained model."
    )
    parser.add_argument("--manifest-pred", "-p", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--do_not_fix_decimals", action="store_true")
    parser.add_argument("--no_cls_and_sep_tokens", action="store_true")
    parser.add_argument(
        "--max_seq_length",
        "-L",
        type=int,
        default=128,
        help="Length of segments into which queries are split. `--max_seq_length` includes [CLS] and [SEP] tokens.",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=8,
        help="Relative shift of consequent segments into which long queries are split. Long queries are split into "
        "segments which can overlap. Parameter `step` controls such overlapping. Imagine that queries are "
        "tokenized into characters, `max_seq_length=5`, and `step=2`. In such a case query 'hello' is tokenized "
        "into segments `[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`.",
    )
    parser.add_argument(
        "--margin",
        "-g",
        type=int,
        default=16,
        help="A number of subtokens in the beginning and the end of segments which output probabilities are not used "
        "for prediction computation. The first segment does not have left margin and the last segment does not have "
        "right margin. For example, if input sequence is tokenized into characters, `max_seq_length=5`, `step=1`, "
        "and `margin=1`, then query 'hello' will be tokenized into segments `[['[CLS]', 'h', 'e', 'l', '[SEP]'], "
        "['[CLS]', 'e', 'l', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`. These segments are passed to the "
        "model. Before final predictions computation, margins are removed. In the next list, subtokens which logits "
        "are not used for final predictions computation are marked with asterisk: `[['[CLS]'*, 'h', 'e', 'l'*, "
        "'[SEP]'*], ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]`.",
    )
    parser.add_argument(
        "--make_queries_contain_intact_sentences",
        action="store_true",
        help="If this option is set, then 1) leading punctuation is removed, 2) first word is made upper case if it is"
        "not yet upper case, 3) if trailing punctuation does not make sentence end, then trailing punctuation is "
        "removed and dot is added.",
    )
    parser.add_argument(
        "--no_all_upper_label",
        action="store_true",
        help="Whether to use 'u' as first character capitalization and 'U' as capitalization of all characters in a "
        "word. If not set, then 'U' is for capitalization of first character in a word, 'O' for absence of "
        "capitalization, 'u' is not used.",
    )
    args = parser.parse_args()
    args.manifest_to_align_with = args.manifest_to_align_with.expanduser()
    args.manifest_pred = args.manifest_pred.expanduser()
    args.output = args.output.expanduser()
    return args


def get_talk_id_order(manifest):
    talk_ids = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            m = TALK_ID_COMPILED_PATTERN.search(data["audio_filepath"])
            if m is None:
                raise ValueError(f"Talk id is not identified in file {manifest} for line {i}")
            talk_ids.append(m.group(0))
    return talk_ids


def load_manifest_text(manifest, text_key):
    result = {}
    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            m = TALK_ID_COMPILED_PATTERN.search(data["audio_filepath"])
            if m is None:
                raise ValueError(f"Talk id is not identified in file {manifest} for line {i}")
            result[m.group(0)] = data[text_key]
    return result


def split_into_segments(text, size):
    segments = []
    for i in range(0, len(text), size):
        segments.append(text[i : i + size])
    return segments


def insert_commas_in_long_numbers(match):
    number = match.group(0)
    result = ""
    count = 0
    for i in range(0, len(number) - 3, 3):
        result = ',' + number[len(number) - i - 3 : len(number) - i] + result
        count += 3
    result = number[: len(number) - count] + result
    return result


def decimal_repl(match):
    text = PUNCTUATION.sub('', match.group(0))
    parts = text.split()
    return parts[0] + '.' + ''.join(parts[2:])


def main():
    args = get_args()
    if args.model in [x.pretrained_model_name for x in PunctuationCapitalizationModel.list_available_models()]:
        model = PunctuationCapitalizationModel.from_pretrained(args.model)
    else:
        model = PunctuationCapitalizationModel.restore_from(os.path.expanduser(args.model))
    order = get_talk_id_order(args.manifest_to_align_with)
    texts_to_process = load_manifest_text(args.manifest_pred, "pred_text")
    texts = [texts_to_process[talk_id] for talk_id in order]
    processed_texts = model.add_punctuation_capitalization(
        texts,
        batch_size=MAX_NUM_SUBTOKENS_IN_INPUT // args.max_seq_length,
        max_seq_length=args.max_seq_length,
        step=args.step,
        margin=args.margin,
        add_cls_and_sep_tokens=not args.no_cls_and_sep_tokens,
    )
    if args.do_not_fix_decimals:
        processed = processed_texts
    else:
        processed = []
        for text in processed_texts:
            processed.append(DECIMAL.sub(decimal_repl, SPACE_DEDUP.sub(' ', text)))
            # processed.append(
            #     LONG_NUMBER.sub(
            #         insert_commas_in_long_numbers,
            #         DECIMAL.sub(decimal_repl, SPACE_DEDUP.sub(' ', ' '.join(processed_segments))),
            #     )
            # )
    if args.make_queries_contain_intact_sentences:
        for i, text in enumerate(processed_texts):
            text = LEFT_PUNCTUATION_STRIP_PATTERN.sub('', text.strip())
            if text[0].islower():
                if args.save_labels_instead_of_text:
                    if text[0] == 'O':
                        text = ('U' if args.no_all_upper_label else 'u') + text[1:]
                else:
                    text = text[0].upper() + text[1:]
            if text[-1] not in '.?!':
                text = RIGHT_PUNCTUATION_STRIP_PATTERN.sub('', text) + '.'
            processed_texts[i] = text
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        for t in processed:
            f.write(t + '\n')


if __name__ == "__main__":
    main()
