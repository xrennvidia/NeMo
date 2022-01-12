import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--isrc", type=Path, required=True)
    parser.add_argument("--itgt", type=Path, required=True)
    parser.add_argument("--osrc", type=Path, required=True)
    parser.add_argument("--otgt", type=Path, required=True)
    args = parser.parse_args()
    for arg_name in ["input_source", "input_target", "output_source", "output_target"]:
        setattr(args, arg_name, getattr(args, arg_name).expanduser())
    return args


def main():
    args = get_args()
    with args.isrc.open() as isf, args.itgt.open() as itf, args.osrc.open('w') as osf, args.otgt.open() as otf:
        for sline, tline in zip(isf, itf):
            if sline and tline:
                osf.write(sline)
                otf.write(tline)


if __name__ == "__main__":
    main()
