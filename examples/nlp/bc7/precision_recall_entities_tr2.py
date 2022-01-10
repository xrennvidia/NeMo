from argparse import ArgumentParser
from typing import List, Tuple
from collections import Counter
import os
import csv


def compute_metrics(predictions: List[str], labels: List[str], separator: str) -> Tuple[float, float, float]:
    """
    Compute the precision, recall and F1 scores of the predictions.
    """
    tp = 0
    fp = 0
    fn = 0

    for prediction, label in zip(predictions, labels):
        prediction_counter = Counter()
        label_counter = Counter()

        # split on the separator so we get the individual entities
        prediction = prediction.split(separator)
        label = label.split(separator)

        # count the number of predictions of each entity
        for pred in prediction:
            prediction_counter[pred] += 1

        # count the number of expected occurence of each entity
        for lab in label:
            label_counter[lab] += 1

        for item in prediction:
            if item in label_counter and label_counter[item] > 0:
                tp += 1
                label_counter[item] -= 1
            else:
                fp += 1

        for item in label:
            if item not in prediction_counter or prediction_counter[item] == 0:
                fn += 1
            else:
                prediction_counter[item] -= 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return round(precision, 3), round(recall, 3), round(f1, 3)


def parse_file(path: str) -> Tuple[List[str], List[str]]:
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

            prediction = prediction_line[prediction_line.find("answers: ") + len("answers: "):].strip()
            label = label_line[
                         label_line.find("answers: ") + len("answers: "): label_line.rfind("<|endoftext|>")].strip()

            predictions.append(prediction)
            labels.append(label)

    if not predictions and not labels:
        raise ValueError("File does not follow expected format.")

    return predictions, labels


def main(source: str, separator: str, out: str, model_label: str, tokens_to_generate: str, overwrite: bool) -> None:
    """
    Parse the prediction data from <source> and write the results to <out>.
    """
    predictions, labels = parse_file(source)
    precision, recall, f1 = compute_metrics(predictions, labels, separator)

    mode = "a" if os.path.exists(out) and not overwrite else "w"

    with open(out, mode) as out_f:
        writer = csv.writer(out_f, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        if mode == "w":
            writer.writerow(["model", "TOKENS_TO_GENERATE", "prec", "rec", "f1"])

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

    args = parser.parse_args()

    main(args.source, args.separator, args.out, args.model_label, args.tokens_to_generate, args.overwrite)
