import argparse
import re
from pathlib import Path


SRCSET = re.compile(r'^<srcset setid="([^"]+)" srclang="([^"]+)">$', flags=re.MULTILINE)
REFSET = re.compile(
    r'^<refset setid="([^"]+)" srclang="([^"]+)" trglang="([^"]+)" refid="([^"]+)">$', flags=re.MULTILINE
)
srcset_tmpl = '<srcset setid="{setid}" srclang="{srclang}">'
refset_tmpl = '<refset setid="{setid}" srclang="{srclang}" trglang="{trglang}" refid="{refid}">'


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--isrc", type=Path, required=True, help="Path to original source file.")
    parser.add_argument("--itgt", type=Path, required=True, help="Path to original reference file.")
    parser.add_argument(
        "--osrc", type=Path, required=True, help="Path to source file after swap. It will contain text from `--isrc`."
    )
    parser.add_argument(
        "--otgt",
        type=Path,
        required=True,
        help="Path to reference file after swap. It will contain text from `--itgt`.",
    )
    args = parser.parse_args()
    for arg_name in ['isrc', 'itgt', 'osrc', 'otgt']:
        setattr(args, arg_name, getattr(args, arg_name).expanduser())
    return args


def main():
    args = get_args()
    with args.isrc.open() as isf, args.itgt.open() as itf:
        source_text = isf.read()
        target_text = itf.read()
    src_line_match = SRCSET.search(source_text)
    if src_line_match is None:
        raise ValueError(f"Source file {args.isrc} does not have <srcset> section matching pattern '{SRCSET.pattern}'")
    ref_line_match = REFSET.search(target_text)
    if ref_line_match is None:
        raise ValueError(f"Target file {args.itgt} does not have <refset> section matching pattern '{REFSET.pattern}'")
    src_setid, ref_setid = src_line_match.group(1), ref_line_match.group(1)
    if src_setid != ref_setid:
        raise ValueError(
            f"'setid' attribute values of <srcset> and <refset> are not equal. 'setid' attribute value in <srcset> in "
            f"{args.isrc} file equals '{ref_setid}', 'setid' attribute value in <refset> in "
            f"{args.itgt} file equals '{ref_setid}'."
        )
    src_srclang, ref_srclang = src_line_match.group(2), ref_line_match.group(2)
    if src_srclang != ref_srclang:
        raise ValueError(
            f"'srclang' attribute values of <srcset> and <refset> are not equal. 'srclang' attribute value in "
            f"<srcset> in file {args.isrc} equals '{src_srclang}', whereas 'srclang' value in <refset> in "
            f"{args.itgt} file equals '{ref_srclang}'"
        )
    trglang = ref_line_match.group(3)
    refid = ref_line_match.group(4)
    src_span = src_line_match.span()
    ref_span = ref_line_match.span()
    new_target_text = (
        source_text[:src_span[0]]
        + refset_tmpl.format(setid=src_setid, srclang=trglang, trglang=src_srclang, refid=refid)
        + '\n'
        + source_text[src_span[1]:]
    )
    new_source_text = (
        target_text[:ref_span[0]]
        + srcset_tmpl.format(setid=src_setid, srclang=trglang)
        + '\n'
        + target_text[ref_span[1]:]
    )
    with args.osrc.open('w') as osf, args.otgt.open('w') as otf:
        osf.write(new_source_text)
        otf.write(new_target_text)


if __name__ == "__main__":
    main()