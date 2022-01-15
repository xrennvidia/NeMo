import argparse
import os
import shutil
from pathlib import Path
from subprocess import PIPE, Popen, run
from typing import List


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", required=True, type=Path)
    parser.add_argument("--part_size", required=True, type=int)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_jobs", type=int)
    args = parser.parse_args()
    for arg_name in ['input_file', 'output_dir']:
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


def split_large_file_into_small_files(input_file: Path, output_dir: Path, part_size: int) -> List[Path]:
    num_lines_in_input_file = count_lines(input_file)
    processes = []
    opened_files = []
    split_files = []
    output_dir.mkdir(parents=True, exist_ok=True)
    num_parts = num_lines_in_input_file // part_size
    for i in range(num_parts):
        start = i * part_size
        new_file = output_dir / f"{i}.txt"
        opened_files.append(new_file.open('w'))
        split_files.append(new_file)
        processes.append(
            Popen(
                [
                    'sed',
                    '-n',
                    f'{start + 1},'
                    f'{start + (part_size if i < num_parts - 1 else num_lines_in_input_file - start)}p',
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


def main() -> None:
    args = get_args()
    split_large_file_into_small_files(args.input_file, args.output_dir, args.part_size)


if __name__ == "__main__":
    main()