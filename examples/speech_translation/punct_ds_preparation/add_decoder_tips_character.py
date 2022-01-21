import argparse
import multiprocessing as mp
import re
import tempfile
from pathlib import Path
from subprocess import PIPE, Popen, run


WORD_TOKENS_PATTERN = re.compile('[uUO]')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        help="A path to autoregressive labels file to which special 'W' is inserted before word token.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="A path to the output file",
    )
    parser.add_argument(
        "--max_number_of_lines_processed_by_one_process",
        type=int,
        default=10 ** 6,
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        help="Default value is number of CPU cores.",
        default=mp.cpu_count()
    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def count_lines_in_file(file_path: Path) -> int:
    result = run(['wc', '-l', str(file_path)], stdout=PIPE, stderr=PIPE)
    if not result:
        raise ValueError(
            f"Bash command `wc -l {file_path}` returned and empty string. "
            f"Possibly, file {file_path} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def main() -> None:
    args = get_args()
    num_lines = count_lines_in_file(args.input)
    if num_lines > args.max_number_of_lines_processed_by_one_process:
        num_splits = 2
        while num_lines // num_splits > args.max_number_of_lines_processed_by_one_process:
            num_splits += 1
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            run(["split", "--number", f"l/{num_splits}", f"{args.input}", f"{tmp_dir_name}/x"], check=True)
            with mp.Pool(args.num_jobs) as pool:
                pool.map(insert_word_character, list(Path(tmp_dir_name).iterdir()))


if __name__ == "__main__":
    main()