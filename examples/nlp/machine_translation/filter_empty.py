import argparse
from pathlib import Path
from subprocess import PIPE, run

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--isrc", type=Path, required=True)
    parser.add_argument("--itgt", type=Path, required=True)
    parser.add_argument("--osrc", type=Path, required=True)
    parser.add_argument("--otgt", type=Path, required=True)
    args = parser.parse_args()
    for arg_name in ["isrc", "itgt", "osrc", "otgt"]:
        setattr(args, arg_name, getattr(args, arg_name).expanduser())
    return args


def count_lines(input_file: Path) -> int:
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    return int(result.stdout.decode('utf-8').split()[0])


def main():
    args = get_args()
    src_num_lines = count_lines(args.isrc)
    tgt_num_lines = count_lines(args.itgt)
    if src_num_lines != tgt_num_lines:
        raise ValueError(
            f"Number of lines {src_num_lines} in file {args.isrc} is not equal to number of lines {tgt_num_lines} in "
            f"file {args.itgt}."
        )
    with args.isrc.open() as isf, args.itgt.open() as itf, args.osrc.open('w') as osf, args.otgt.open('w') as otf:
        for sline, tline in tqdm(zip(isf, itf), total=src_num_lines, desc="Filtering empty lines", unit="line"):
            if sline.strip() and tline.strip():
                osf.write(sline)
                otf.write(tline)


if __name__ == "__main__":
    main()
