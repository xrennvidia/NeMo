import argparse
from pathlib import Path

from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE

from tts_asr import asr_worker


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda_device", required=True, type=int)
    parser.add_argument("--world_size", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--start_line", required=True, type=int)
    parser.add_argument("--num_lines", required=True, type=int)
    parser.add_argument(
        "--asr_model",
        required=True,
        choices=[
            x.pretrained_model_name
            for x in EncDecCTCModel.list_available_models() + EncDecCTCModelBPE.list_available_models()
        ],
    )
    parser.add_argument("--asr_batch_size", required=True, type=int)
    parser.add_argument("--tmp_wav_dir", required=True, type=Path)
    parser.add_argument("--tmp_txt_dir", required=True, type=Path)
    args = parser.parse_args()
    args.tmp_wav_dir = args.tmp_wav_dir.expanduser()
    args.tmp_txt_dir = args.tmp_txt_dir.expanduser()
    return args


def main() -> None:
    args = get_args()
    asr_worker(
        args.rank,
        args.world_size,
        args.cuda_device,
        args.asr_model,
        args.asr_batch_size,
        args.tmp_wav_dir,
        args.tmp_txt_dir,
        args.start_line,
        args.num_lines,
    )


if __name__ == "__main__":
    main()
