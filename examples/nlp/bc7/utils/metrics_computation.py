from typing import *
from collections import Counter


def compute_metrics(predictions: List[str], labels: List[str], separator: str, verbose: bool=False) -> Tuple[float, float, float]:
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
        prediction = prediction.lower().split(separator)
        label = label.lower().split(separator)

        if verbose:
            print("Prediction:", prediction)
            print("Label:", label)

        # count the number of predictions of each entity
        for pred in prediction:
            prediction_counter[pred] += 1

        # count the number of expected occurence of each entity
        for lab in label:
            label_counter[lab] += 1

        if prediction == [""]:
            fn += sum(label_counter.values())
            if verbose:
                print("**************************\n")
            continue

        if verbose:
            print(prediction_counter)
            print(label_counter)
        for item in prediction:
            if item and item in label_counter and label_counter[item] > 0:
                tp += 1
                if verbose:
                    print("TP +1")
                label_counter[item] -= 1
            else:
                fp += 1
                if verbose:
                    print("FP +1")

        for item in label:
            if item not in prediction_counter or prediction_counter[item] == 0:
                fn += 1
                if verbose:
                    print("FN +1")
            else:
                prediction_counter[item] -= 1

        if verbose:
            print("**************************\n")

    if verbose:
        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return round(precision, 3), round(recall, 3), round(f1, 3)