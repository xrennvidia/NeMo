import argparse
from pathlib import Path

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--tts_model_spectrogram",
        help="Name of TTS spectrogram model.",
        required=True,
        choices=[x.pretrained_model_name for x in SpectrogramGenerator.list_available_models()],
    )
    parser.add_argument(
        "--tts_model_vocoder",
        help="Name of TTS vocoder model.",
        required=True,
        choices=[x.pretrained_model_name for x in Vocoder.list_available_models()],
    )
    parser.add_argument(
        "--asr_model",
        help="Name of ASR CTC char model.",
        required=True,
        choices=[x.pretrained_model_name for x in EncDecCTCModel.list_available_models()]
    )
    parser.add_argument(
        "--input",
        help="Path to input text file in English, e.g. English part of WMT.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Path to output text file with uncased augmented text",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--num_lines_in_chunk",
        help="Number of lines loaded in memory for processing. Loaded lines are transformed into audio and then "
        "back to text.",
        type=int,
        default=5 * 10 ** 4,
    )
    parser.add_argument(
        "--tmp_wav_files_dir",
        type=Path,
        required=True,
        help="Path to a directory where temporary .wav files will be saved.",
        default=Path("_tmp_wav_files"),
    )
    parser.add_argument(
        "--num_lines_for_1_iteration",
        type=int,
        help="Number of lines in `--input` file which will be augmented. Lines in the beginning of the file are used."
        "Be default all lines are augmented.",
    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    args.tmp_wav_files_dir = args.tmp_wav_files_dir.expanduser()
    return args


def main():
    args = get_args()
    tts_model_spectrogram = SpectrogramGenerator.from_pretrained(args.tts_model_spectrogram)
    tts_model_vocoder = Vocoder.from_pretrained(args.tts_model_vocoder)
    asr_model = EncDecCTCModel.from_pretrained(args.asr_model)


if __name__ == "__main__":
    main()