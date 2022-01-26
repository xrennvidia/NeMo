import argparse
from copy import deepcopy
from pathlib import Path

import torch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--old_style_model_weights_ckpt", type=Path, required=True)
    parser.add_argument("--reference_new_style_model_weights_ckpt", type=Path, required=True)
    parser.add_argument("--output_model_weights_ckpt", type=Path, required=True)
    args = parser.parse_args()
    for arg_name in [
        'old_style_model_weights_ckpt', 'reference_new_style_model_weights_ckpt', 'output_model_weights_ckpt'
    ]:
        setattr(args, arg_name, getattr(args, arg_name).expanduser())
    return args


def main() -> None:
    args = get_args()
    input_state_dict = torch.load(args.old_style_model_weights_ckpt)
    reference_state_dict = torch.load(args.reference_new_style_model_weights_ckpt)
    output_state_dict = deepcopy(reference_state_dict)
    for k, v in input_state_dict.items():
        output_state_dict[k] = v
    output_state_dict['decoder.replacement_token_embedding.weight'] = input_state_dict[
        'encoder._embedding.token_embedding.weight'
    ]


if __name__ == "__main__":
    main()