import json
import re
from argparse import ArgumentParser
from subprocess import run, PIPE
from pathlib import Path

from tqdm import tqdm


BUFFER_SIZE = 2 ** 24


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


TEXT_TO_NUMBERS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "twenty one": "21",
    "twenty two": "22",
    "twenty three": "23",
    "twenty four": "24",
    "twenty five": "25",
    "twenty six": "26",
    "twenty seven": "27",
    "twenty eight": "28",
    "twenty nine": "29",
    "thirty": "30",
    "thirty one": "31",
    "thirty two": "32",
    "thirty three": "33",
    "thirty four": "34",
    "thirty five": "35",
    "thirty six": "36",
    "thirty seven": "37",
    "thirty eight": "38",
    "thirty nine": "39",
    "forty": "40",
    "forty one": "41",
    "forty two": "42",
    "forty three": "43",
    "forty four": "44",
    "forty five": "45",
    "forty six": "46",
    "forty seven": "47",
    "forty eight": "48",
    "forty nine": "49",
    "fifty": "50",
    "fifty one": "51",
    "fifty two": "52",
    "fifty three": "53",
    "fifty four": "54",
    "fifty five": "55",
    "fifty six": "56",
    "fifty seven": "57",
    "fifty eight": "58",
    "fifty nine": "59",
    "sixty": "60",
    "sixty one": "61",
    "sixty two": "62",
    "sixty three": "63",
    "sixty four": "64",
    "sixty five": "65",
    "sixty six": "66",
    "sixty seven": "67",
    "sixty eight": "68",
    "sixty nine": "69",
    "seventy": "70",
    "seventy one": "71",
    "seventy two": "72",
    "seventy three": "73",
    "seventy four": "74",
    "seventy five": "75",
    "seventy six": "76",
    "seventy seven": "77",
    "seventy eight": "78",
    "seventy nine": "79",
    "eighty": "80",
    "eighty one": "81",
    "eighty two": "82",
    "eighty three": "83",
    "eighty four": "84",
    "eighty five": "85",
    "eighty six": "86",
    "eighty seven": "87",
    "eighty eight": "88",
    "eighty nine": "89",
    "ninety": "90",
    "ninety one": "91",
    "ninety two": "92",
    "ninety three": "93",
    "ninety four": "94",
    "ninety five": "95",
    "ninety six": "96",
    "ninety seven": "97",
    "ninety eight": "98",
    "ninety nine": "99",
}


def add_ordinals_to_numbers():
    for k, v in TEXT_TO_NUMBERS.copy().items():
        if k.endswith("one"):
            TEXT_TO_NUMBERS[k[:-3] + "first"] = v + 'st'
        elif k.endswith("two"):
            TEXT_TO_NUMBERS[k[:-3] + "second"] = v + 'nd'
        elif k.endswith("three"):
            TEXT_TO_NUMBERS[k[:-5] + "third"] = v + "rd"
        elif k.endswith("five"):
            TEXT_TO_NUMBERS[k[:-4] + "fifth"] = v + "th"
        elif k.endswith("eight"):
            TEXT_TO_NUMBERS[k + 'h'] = v + 'th'
        elif k.endswith("nine"):
            TEXT_TO_NUMBERS[k[:-4] + "ninth"] = v + 'th'
        elif k.endswith("twelve"):
            TEXT_TO_NUMBERS[k[:-6] + "twelfth"] = v + 'th'
        elif k.endswith("y"):
            TEXT_TO_NUMBERS[k[:-1] + 'ieth'] = v + 'th'
        else:
            TEXT_TO_NUMBERS[k + 'th'] = v + 'th'


add_ordinals_to_numbers()


def add_hyphen_numbers():
    global TEXT_TO_NUMBERS
    new_text_to_numbers = {}
    for k, v in TEXT_TO_NUMBERS.copy().items():
        if any(
            [
                k.startswith(start)
                for start in ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
            ]
        ) and any(
            [
                k.endswith(end)
                for end in [
                    'one',
                    'two',
                    'three',
                    'four',
                    'five',
                    'six',
                    'seven',
                    'eight',
                    'nine',
                    'first',
                    'second',
                    'third',
                    'fourth',
                    'fifth',
                    'sixth',
                    'seventh',
                    'eighth',
                    'ninth',
                ]
            ]
        ):
            new_text_to_numbers[k] = v
            new_text_to_numbers[k.replace(' ', '-')] = v
        else:
            new_text_to_numbers[k] = v
    TEXT_TO_NUMBERS = new_text_to_numbers


add_hyphen_numbers()


def add_decades_to_numbers():
    decades = {
        "twenties": "20s",
        "thirties": "30s",
        "forties": "40s",
        "fifties": "50s",
        "sixties": "60s",
        "seventies": "70s",
        "eighties": "80s",
        "nineties": "90s",
    }
    TEXT_TO_NUMBERS.update(decades)


add_decades_to_numbers()


def flip_dict(d):
    result = {}
    for k, v in d.items():
        if v in result:
            raise ValueError(f"Flipped dict has to have unique values. Value {repr(v)} is repeated at least twice.")
        result[v] = k
    return result


SINGLE_NUMBERS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
SINGLE_ORDINALS = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "forth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "nineth",
}


def str_to_number_repl(match):
    return TEXT_TO_NUMBERS[match.group(0).lower()]


def single_number_to_str_repl(match):
    return SINGLE_NUMBERS[match.group(0).lower()]


def single_ordinal_to_str_repl(match):
    return SINGLE_ORDINALS[match.group(0)]


def hundred_repl(match_obj):
    if match_obj.group(2) is None:
        second_term = 0
    else:
        t = match_obj.group(2)[5:] if match_obj.group(2).startswith(" and ") else match_obj.group(2)
        second_term = int(t)
    return str(int(match_obj.group(1)) * 100 + second_term)


def ten_power_3n_repl(match_obj):
    # number_groups = [1, 6, 10, 13]
    number_groups = [1, 4]
    for i_ng, ng in enumerate(number_groups):
        if match_obj.group(ng) is not None:
            start = ng + 1
            # n = 4 - i_ng
            n = 2 - i_ng
            break
    result = 0
    for i in range(start, start + n):
        power = (start + n - i - 1) * 3
        if match_obj.group(i) is None:
            term = 0
        else:
            t = match_obj.group(i)[5:] if match_obj.group(i).startswith(" and ") else match_obj.group(i)
            term = int(t)
        result += term * 10 ** power
    return str(result)


def month_day_repl(match):
    return match.group(1) + ' ' + match.group(2)[:-2]


def decimal_repl(match):
    text = match.group(0)
    parts = text.split()
    return parts[0] + '.' + ''.join(parts[2:])


def decimal_deparse_repl(match):
    text = match.group(0)
    int_part = text.split('.')[0]
    fraction_part = text.split('.')[1]
    return int_part + ' point ' + ' '.join(fraction_part)


def oh_repl(match):
    return '0' + TEXT_TO_NUMBERS[match.group(1)]


REPLACEMENTS = [
    (
        re.compile(
            r'oh \b(' + '|'.join([rf'\b{str_num}\b' for str_num in list(TEXT_TO_NUMBERS.keys())[1:10]]) + r')\b'
        ),
        oh_repl,
    ),
    (
        re.compile('|'.join([rf'\b{str_num}\b' for str_num in list(TEXT_TO_NUMBERS.keys())[::-1]]), flags=re.I),
        str_to_number_repl,
    ),
    (re.compile(r"\b([1-9]|1[1-9]) hundred((?: and)? [1-9][0-9]?)?", flags=re.I), hundred_repl),
    (
        re.compile(
            # r"(\b(?:([1-9][0-9]{0,2}) billion)(?:( [1-9][0-9]{0,2}) million)?(?:( [1-9][0-9]{0,2}) thousand)?"
            # r"( [1-9][0-9]{0,2})?)|"
            # r"(\b(?:([1-9][0-9]{0,2}) million)(?:( [1-9][0-9]{0,2}) thousand)?( [1-9][0-9]{0,2})?)|"
            r"(\b(?:([1-9][0-9]{0,2}) thousand)((?: and)? [1-9][0-9]{0,2})?)|(\b([1-9][0-9]{0,2}))",
            flags=re.IGNORECASE,
        ),
        ten_power_3n_repl,
    ),
    (re.compile(r"(?<![0-9] )\b([12][0-9]) ([0-9]{2})(?! [0-9])", flags=re.IGNORECASE), r"\1\2"),
    (
        re.compile(r"(?:[0-9]+) point(?: [0-9])+", flags=re.I),
        decimal_repl,
    ),  # before replacing single digits parse decimals
    (re.compile(r"(?<!\.)\b[0-9]\b(?!\.)", flags=re.I), single_number_to_str_repl),
    (re.compile(r"(?:[1-9][0-9]*|0)\.[0-9]*[1-9]"), decimal_deparse_repl),
    (re.compile(r"\s+", flags=re.I), " "),
    (
        re.compile(
            f'({"|".join(MONTHS)})'
            + ' ('
            + "|".join([rf"\b{k}\b" for k in list(TEXT_TO_NUMBERS.values())[212:172:-1]])  # ordinals from 31st to 1st
            + ')',
            flags=re.I,
        ),
        month_day_repl,
    ),
    (
        re.compile(
            rf"\b(?:{'|'.join(list(TEXT_TO_NUMBERS.values())[173:182])})\b(?! of ({'|'.join(MONTHS)})\b)", flags=re.I
        ),
        single_ordinal_to_str_repl,
    ),
]


def text_to_numbers(text):
    for r in REPLACEMENTS:
        text = r[0].sub(r[1], text)
    return text


def count_lines(input_file: Path) -> int:
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    if not result:
        raise ValueError(
            f"Bash command `wc -l {input_file}` returned and empty string. "
            f"Possibly, file {input_file} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def get_args():
    parser = ArgumentParser()
    input_ = parser.add_mutually_exclusive_group(required=True)
    input_.add_argument("--input", "-i", help="Path to input manifest file.", type=Path)
    input_.add_argument("--input_text", help="Path to input text file", type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", "-o", help="Path to output manifest file.", type=Path)
    output.add_argument("--output_text", help="Path to output text file. ", type=Path)
    parser.add_argument("--text-key", "-k", help="Text key in manifest. Default is `pred_text`.", default="pred_text")
    args = parser.parse_args()
    if args.output is not None and args.input is None:
        parser.error(
            f"If you provide parameter `--output` you also have to provide parameter `--input`. "
            f"`--output={args.output}`."
        )
    for arg_name in ['input', 'input_text', 'output', 'output_text']:
        arg = getattr(args, arg_name)
        if arg is not None:
            setattr(args, arg_name, arg.expanduser())
    return args


def main():
    args = get_args()
    input_file = args.input_text if args.input is None else args.input
    output_file = args.output_text if args.output is None else args.output
    with input_file.open(buffering=BUFFER_SIZE) as in_f, output_file.open('w', buffering=BUFFER_SIZE) as out_f:
        for line in tqdm(in_f, total=count_lines(input_file), desc="Transforming text to numbers", unit='line'):
            if args.input is None:
                out_text = text_to_numbers(line)
            else:
                in_data = json.loads(line)
                out_text = text_to_numbers(in_data[args.text_key])
            if args.output is None:
                out_f.write(out_text + ('\n' if out_text[-1] != '\n' else ''))
            else:
                in_data[args.text_key] = out_text
                out_f.write(json.dumps(in_data) + '\n')


if __name__ == "__main__":
    main()
