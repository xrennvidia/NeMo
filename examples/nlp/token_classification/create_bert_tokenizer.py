import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--train_text", required=True, type=Path, help="Path to text file used for tokenizer training."
    )
    parser.add_argument("--vocab_size", type=int, help="Number of tokens in tokenizer vocabulary.", default=25000)
    parser.add_argument("--output_file", required=True, type=Path, help="Path to the output JSON file with the model.")
    args = parser.parse_args()
    args.train_text = args.train_text.expanduser()
    args.output_file = args.output_file.expanduser()
    return args


def main() -> None:
    args = get_args()
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.normalizer = BertNormalizer()
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.train(trainer=trainer, files=[args.train_text])
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.save(args.output)


if __name__ == "__main__":
    main()