# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from argparse import ArgumentParser
from typing import List, Tuple
from collections import Counter
import os
import csv
from utils.metrics_computation import compute_metrics


"""
Usage:

python precision_recall_entities_tr2.py \
    --source <path/to/entities/inference/file> \
    --out <path/to/csv/output/location> \
    --model_label <label for recorded results> \
    --separator <separator used (if any) between the predicted entities> 

If the ending prompt is different from the default one used in the argument parsing here, provide it using the 
--prompt_end flag.
"""


def parse_file(path: str, prompt_end: str) -> Tuple[List[str], List[str]]:
    """
    Read the given path of a text file and return a  list of predictions and labels.
    """
    data = open(path, "r")
    lines = data.readlines()
    data.close()

    predictions = []
    labels = []

    for i, line in enumerate(lines):
        if line.startswith("Completed"):
            prediction_line = line
            label_line = lines[i + 2]

            prediction = prediction_line[prediction_line.find(prompt_end) + len(prompt_end):].strip()
            label = label_line[
                         label_line.find(prompt_end) + len(prompt_end): label_line.rfind("<|endoftext|>")].strip()    #  label_line.rfind("<|endoftext|>")

            predictions.append(prediction)
            labels.append(label)

    if not predictions and not labels:
        raise ValueError("File does not follow expected format.")

    return predictions, labels


def main(source: str, separator: str, out: str, model_label: str, tokens_to_generate: str, overwrite: bool, verbose: bool, prompt_end: str) -> None:
    """
    Parse the prediction data from <source> and write the results to <out>.
    """
    predictions, labels = parse_file(source, prompt_end)
    precision, recall, f1 = compute_metrics(predictions, labels, separator, verbose)

    mode = "a" if os.path.exists(out) and not overwrite else "w"

    with open(out, mode) as out_f:
        writer = csv.writer(out_f, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        if mode == "w":
            writer.writerow(["model", "TOKENS_TO_GENERATE", "prec", "rec", "f1"])

        print("Label:", model_label)
        print("Number of tokens to generate:", tokens_to_generate)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        writer.writerow([model_label, tokens_to_generate, precision, recall, f1])
        out_f.flush()


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute the precision, recall and F1 score for the named entity recognition"
                                        "task done on the BC7 Track 2 data.")

    parser.add_argument(
        "--source",
        required=True,
        type=str,
        help="Path to file containing predictions and labels."
    )

    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Path to a .csv file to save results to."
    )

    parser.add_argument(
        "--model_label",
        required=True,
        type=str,
        help="A string to denote the name of the model used for inference to report in the scores file."
    )

    parser.add_argument(
        "--separator",
        type=str,
        default=", ",
        help="The string separating each value in the predictions and labels."
    )

    parser.add_argument(
        "--prompt_end", type=str, default="answers:", required=False,
        help="The ending prompt set up for NER inference that signals where the model should start generating text."
    )

    parser.add_argument(
        "--tokens_to_generate",
        type=str,
        default="64",
        help="The number of tokens set to generate in the inference to report in the scores file."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Use this argument to overwrite existing output file if it already exists."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use this flag if output logging is desired."
    )

    args = parser.parse_args()

    main(args.source, args.separator, args.out, args.model_label, args.tokens_to_generate, args.overwrite, args.verbose, args.prompt_end)
