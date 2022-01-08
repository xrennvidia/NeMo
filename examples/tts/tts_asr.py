import argparse
import asyncio
import multiprocessing as mp
import shutil
from pathlib import Path
from subprocess import run, PIPE
from typing import List, Tuple

import soundfile
import torch
import torch.multiprocessing as tmp
from torch.utils.data import DataLoader, Dataset

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import Progress
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from nemo.utils import logging
from nemo_text_processing.text_normalization.normalize import Normalizer


TTS_PARSING_REPORT_PERIOD = 100
TTS_SPECTROGRAM_VOCODER_SWITCH_PERIOD = 200


def get_args() -> argparse.Namespace:
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
        "--tmp_dir",
        type=Path,
        required=True,
        help="Path to a directory where temporary .wav and .txt files will be saved.",
        default=Path("_tmp_wav_files"),
    )
    parser.add_argument(
        "--num_lines_per_process_for_1_iteration",
        type=int,
        help="Number of lines in `--input` file which will be augmented. Lines in the beginning of the file are used."
        "Be default all lines are augmented.",
        default=25 * 10 ** 3,
    )
    parser.add_argument(
        "--tts_tokens_in_batch",
        type=int,
        default=50000,
        help="Number of phone tokens in a batch passed for TTS.",
    )
    parser.add_argument(
        "--asr_batch_size",
        type=int,
        default=40,
        help="Number of phone tokens in a batch passed for TTS.",
    )
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=[0, 1], help="List of CUDA devices used for training."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of jobs used for Inverse text normalization. Be default `--n_jobs` parameter is equal to number "
        "of CPU cores",
    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    args.tmp_dir = args.tmp_dir.expanduser()
    return args


class TTSDataset(Dataset):
    def __init__(
        self,
        lines: List[str],
        start_line: int,
        tts_model_spectrogram: SpectrogramGenerator,
        tts_tokens_in_batch: int,
        tts_parsing_progress_queue: mp.Queue,
    ):
        logging.info(f"Looking for start line.")
        tokenized_lines_with_indices = []
        for i, line in enumerate(lines, start=start_line):
            if i % TTS_PARSING_REPORT_PERIOD == 0:
                tts_parsing_progress_queue.put(min(TTS_PARSING_REPORT_PERIOD, i - start_line))
            tokenized_lines_with_indices.append((tts_model_spectrogram.parse(line), i))
        tokenized_lines_with_indices = sorted(tokenized_lines_with_indices, key=lambda x: x[0].shape[1])
        self.batches = []
        batch = []
        current_length = tokenized_lines_with_indices[0][0].shape[1]
        for line_and_i in tokenized_lines_with_indices:
            if (
                line_and_i[0].shape[1] == current_length
                and len(batch) * current_length < tts_tokens_in_batch - current_length
            ):
                batch.append(line_and_i)
            else:
                self.batches.append(batch)
                batch = [line_and_i]
                current_length = line_and_i[0].shape[1]
        if batch:
            self.batches.append(batch)

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int]]:
        batch = self.batches[idx]
        return torch.cat([element[0] for element in batch]), [element[1] for element in batch]


def count_lines(input_file: Path) -> int:
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    return int(result.stdout.decode('utf-8').split()[0])


def get_start_and_num_lines(num_lines: int, rank: int, world_size: int) -> Tuple[int, int]:
    _num_lines_per_process_for_1_iteration = num_lines // world_size
    slice_start = rank * _num_lines_per_process_for_1_iteration
    if rank < world_size - 1:
        num_lines_to_process = _num_lines_per_process_for_1_iteration
    else:
        num_lines_to_process = num_lines - slice_start
    return slice_start, num_lines_to_process


def tts_worker(
    rank: int,
    args: argparse.Namespace,
    lines: List[str],
    start_line: int,
    tts_parsing_progress_queue: mp.Queue,
    tts_progress_queue: mp.Queue,
) -> None:
    slice_start, num_lines_to_process = get_start_and_num_lines(len(lines), rank, len(args.cuda_devices))
    device = torch.device(f'cuda:{args.cuda_devices[rank]}')
    tts_model_spectrogram = SpectrogramGenerator.from_pretrained(args.tts_model_spectrogram, map_location=device).eval()
    vocoder = Vocoder.from_pretrained(args.tts_model_vocoder, map_location=device).eval()
    text_dataset = TTSDataset(
        lines[slice_start : slice_start + num_lines_to_process],
        start_line + slice_start,
        tts_model_spectrogram,
        args.tts_tokens_in_batch,
        tts_parsing_progress_queue,
    )
    accumulated_specs, accumulated_indices = [], []
    for batch_i, (batch_tensor, indices) in enumerate(text_dataset):
        specs = tts_model_spectrogram.generate_spectrogram(tokens=batch_tensor.to(device))
        accumulated_specs.append(specs)
        accumulated_indices.append(indices)
        if (batch_i + 1) % TTS_SPECTROGRAM_VOCODER_SWITCH_PERIOD == 0:
            for _specs, _indices in zip(accumulated_specs, accumulated_indices):
                audio = vocoder.convert_spectrogram_to_audio(spec=_specs)
                for aud, i in zip(audio, _indices):
                    soundfile.write(args.tmp_dir / f"{i}.proc{rank}.wav", aud.cpu(), samplerate=22050)
            accumulated_specs, accumulated_indices = [], []
        tts_progress_queue.put(len(indices))
    if accumulated_specs:
        for _specs, _indices in zip(accumulated_specs, accumulated_indices):
            audio = vocoder.convert_spectrogram_to_audio(spec=_specs)
            for aud, i in zip(audio, _indices):
                soundfile.write(args.tmp_dir / f"{i}.proc{rank}.wav", aud.cpu(), samplerate=22050)


def asr_worker(
    rank: int,
    world_size: int,
    cuda_device: int,
    asr_model: str,
    batch_size: int,
    tmp_dir: Path,
    start_line: int,
    num_lines: int,
) -> None:
    slice_start, num_lines_to_process = get_start_and_num_lines(num_lines, rank, world_size)
    device = torch.device(f'cuda:{cuda_device}')
    asr_model = EncDecCTCModel.from_pretrained(asr_model, map_location=device).eval()
    audio_files = [
        file
        for file in tmp_dir.iterdir()
        if file.suffixes == [f'.proc{rank}', '.wav'] and file.is_file()
    ]
    audio_files = sorted(audio_files, key=lambda x: int(x.stem.split('.')[0]))
    hypotheses = asr_model.transcribe([str(file) for file in audio_files], batch_size=batch_size)
    for file in audio_files:
        file.unlink()
    output_file = tmp_dir / f'{start_line + slice_start}_{num_lines_to_process}.txt'
    with output_file.open('w') as f:
        for h in hypotheses:
            f.write(h + '\n')


def check_and_sort_text_files(files: List[Path], num_lines: int) -> List[Path]:
    files = sorted(files, key=lambda x: int(x.stem.split('_')[0]))
    detected_num_lines = 0
    for i, file in enumerate(files):
        nl = int(file.stem.split('_')[1])
        if i < len(files) - 1:
            assert int(file.stem.split('_')[0]) + nl == int(files[i + 1].stem.split('_')[0]), (
                f"i={i} file={file} files[{i+1}]={files[i+1]} nl={nl}"
            )
        detected_num_lines += nl
    assert detected_num_lines == num_lines, f"detected_num_lines={detected_num_lines} num_lines={num_lines}"
    return files


def unite_text_files(tmp_dir: Path, output: Path, num_lines: int) -> None:
    text_files = [file_path for file_path in tmp_dir.iterdir() if file_path.suffix == '.txt']
    text_files = check_and_sort_text_files(text_files, num_lines)
    with output.open('w') as out_f:
        for text_file in text_files:
            with text_file.open() as tf:
                out_f.write(tf.read())


async def run_asr(cuda_device: int, rank: int, start_line: int, num_lines: int, args: argparse.Namespace) -> None:
    proc = await asyncio.create_subprocess_shell(
        f"python asr_1_process.py "
        f"--cuda_device {cuda_device} "
        f"--world_size {len(args.cuda_devices)} "
        f"--rank {rank} "
        f"--start_line {start_line} "
        f"--num_lines {num_lines} "
        f"--asr_model {args.asr_model} "
        f"--asr_batch_size {args.asr_batch_size} "
        f"--tmp_dir {args.tmp_dir}"
    )
    await proc.communicate()


async def main() -> None:
    args = get_args()
    world_size = torch.cuda.device_count()
    if any([d >= world_size for d in args.cuda_devices]):
        raise ValueError(
            f"Some values of `--cuda_devices` argument are greater or equal than number of GPUs {world_size}. "
            f"Devices: {args.cuda_devices}."
        )
    num_lines = count_lines(args.input)
    if args.tmp_dir.is_dir():
        shutil.rmtree(args.tmp_dir)
    elif args.tmp_dir.is_file():
        args.tmp_dir.unlink()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    normalizer = Normalizer(input_case='cased', lang='en')
    with Progress(num_lines, ["TTS parsing", "TTS"], 'line') as progress_queues:
        with args.input.open() as f:
            start_line = 0
            while True:
                lines = []
                for i in range(args.num_lines_per_process_for_1_iteration * len(args.cuda_devices)):
                    line = f.readline()
                    if line:
                        lines.append(line)
                    else:
                        break
                if not lines:
                    break
                assert all(lines)
                lines = normalizer.normalize_list_parallel(lines, verbose=False, n_jobs=args.n_jobs)
                assert isinstance(lines, list) and all([isinstance(line, str) for line in lines])
                tmp.spawn(
                    tts_worker,
                    args=(args, lines, start_line, progress_queues[0], progress_queues[1]),
                    nprocs=len(args.cuda_devices),
                    join=True,
                )
                await asyncio.gather(
                    *[
                        run_asr(cuda_device, rank, start_line, len(lines), args)
                        for rank, cuda_device in enumerate(args.cuda_devices)
                    ]
                )
                start_line += len(lines)
    logging.info("Uniting text files...")
    unite_text_files(args.tmp_dir, args.output, num_lines)


if __name__ == "__main__":
    asyncio.run(main())
