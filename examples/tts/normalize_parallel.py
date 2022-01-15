import argparse
import os
import shutil
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired, run
from typing import List


NUM_SECONDS_TO_WAIT_FOR_NORM_PROCESS = 0.1


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", required=True, type=Path)
    parser.add_argument("--tmp_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--num_jobs", type=int)
    args = parser.parse_args()
    for arg_name in ['input_file', 'tmp_dir', 'output_file']:
        setattr(args, arg_name, getattr(args, arg_name).expanduser())
    if args.num_jobs is None:
        args.num_jobs = os.cpu_count()
    return args


def count_lines(input_file: Path) -> int:
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    if not result:
        raise ValueError(
            f"Bash command `wc -l {input_file}` returned and empty string. "
            f"Possibly, file {input_file} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def split_large_file_into_small_files(input_file: Path, output_dir: Path, num_lines_per_file: int) -> List[Path]:
    num_lines_in_input_file = count_lines(input_file)
    processes = []
    opened_files = []
    split_files = []
    for i, start in enumerate(range(0, num_lines_in_input_file, num_lines_per_file)):
        new_file = output_dir / f"{i}.txt"
        opened_files.append(new_file.open('w'))
        split_files.append(new_file)
        processes.append(
            Popen(
                [
                    'sed',
                    '-n',
                    f'{start},{start + min(num_lines_per_file, num_lines_in_input_file - start) - 1}p',
                    str(input_file),
                ],
                stdout=opened_files[-1],
            )
        )
    for proc in processes:
        proc.wait()
    for f in opened_files:
        f.close()
    return split_files


def is_int(s):
    try:
        int(s)
    except ValueError:
        return False
    return True


def run_normalization(split_files: List[Path], norm_dir: Path) -> List[Path]:
    output_files, processes = [], []
    for file in split_files:
        output_file = norm_dir / file.name
        processes.append(
            Popen(
                f"python normalize_1_process.py "
                f"--input_file {file} "
                f"--output_file {output_file}".split()
            )
        )
        output_files.append(output_file)
    return_codes = [None] * len(processes)
    while any([c is None for c in return_codes]):
        for i, proc in enumerate(processes):
            try:
                code = proc.wait(NUM_SECONDS_TO_WAIT_FOR_NORM_PROCESS)
            except TimeoutExpired:
                code = None
                pass
            if code is not None and code != 0:
                raise RuntimeError(f"An ASR process number {i} terminated with non zero return code {code}.")
            return_codes[i] = code
    return output_files


def unite_text_files(normalized_files: List[Path], output: Path) -> None:
    with output.open('w') as out_f:
        for file in normalized_files:
            with file.open() as tf:
                out_f.write(tf.read())


def main() -> None:
    args = get_args()
    if args.tmp_dir.is_file():
        args.tmp_dir.unlink()
    elif args.tmp_dir.is_dir():
        shutil.rmtree(str(args.tmp_dir))
    split_dir = args.tmp_dir / "split"
    norm_dir = args.tmp_dir / "norm"
    num_lines = count_lines(args.input_file)
    num_lines_per_file = num_lines // args.num_jobs
    split_files = split_large_file_into_small_files(args.input_file, split_dir, num_lines_per_file)
    normalized_files = run_normalization(split_files, norm_dir)
    unite_text_files(normalized_files, args.output_file)


if __name__ == "__main__":
    main()
