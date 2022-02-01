from typing import Set, List, Any, Tuple
from utils.utils_bc7tr2 import parse_data, PaperDocument
from typing import *
from collections import defaultdict
from tqdm.auto import tqdm
import re
from string import punctuation
import os
import json
from argparse import ArgumentParser


class Entity:
    def __init__(self, text: str) -> None:
        self.full_text = text

        self.num_words = len(words)
        self.first_word = words[0]
        self.remaining = words[1:]
        self.len_remaining = len(self.remaining)

    def matches_entity(self, beginning: str, remaining: List[str]) -> bool:
        """
        Using the given beginning word and the remaining words to return True if the full string matches with this
        current entity, and False otherwise.
        """
        beginning = beginning.strip(punctuation)
        remaining = [word.strip(punctuation) for word in remaining]
        return beginning == self.first_word and remaining == self.remaining


def get_targets(documents: List[Dict[str, Any]], extract_topics: bool = False) \
        -> Union[List[Tuple[Any, List[str]]], List[Tuple[Any, List[str], List[str]]]]:
    """
    Extract the named entities that should be recognized from each document's text, and create the labels for it.

    If topics is False, the labels are simply all the named entities (all chemicals). Otherwise, only the topic
    chemicals will be given.
    """
    data = []

    for i, document in tqdm(enumerate(documents), desc="Processing NER targets..."):
        doc_class = PaperDocument(document)

        # get the abstract document -- should be the second passage
        abstract_text = doc_class.data.passages[1].text
        annotations = doc_class.data.passages[1].annotations

        topics = []
        expected_entities = []

        for annotation in annotations:
            if annotation.infons.type == "MeSH_Indexing_Chemical":
                if extract_topics:
                    topics.append(annotation.infons.entry_term)
                expected_entities.append(annotation.infons.entry_term)

            elif annotation.text != "":
                expected_entities.append(annotation.text)

        if not extract_topics:
            data.append((abstract_text, expected_entities))
        else:
            data.append((abstract_text, topics, expected_entities))

    return data


def load_and_write_data(input_path: str, output_dir: str, base_output_name: str, prompt_start: str, prompt_end: str,
                        separator: str) -> None:
    """
    Load the data from the path and write the output to the desired output directory.
    """
    data_corpus = parse_data(input_path)
    data_with_targets = get_targets(data_corpus)

    output_data_file = open(os.path.join(output_dir, f"{base_output_name}_full_text.json"), "w+")

    for entry in data_with_targets:
        inputs, labels = entry

        labels_text = separator.join(f"{item}" for item in labels if item != "")

        text = f"<|endoftext|> {prompt_start} {inputs} {prompt_end} {labels_text} <|endoftext|>"
        re.sub(" {2,}", " ", text)
        data_entry = {"text": text}

        json.dump(data_entry, output_data_file)
        output_data_file.write("\n")

    output_data_file.close()


def load_and_write_topics(input_path: str, output_dir: str, base_output_name: str, prompt_start: str, prompt_end: str,
                          separator: str, topics_with_abstract: bool) -> None:
    data_corpus = parse_data(input_path)
    data_with_topics = get_targets(data_corpus, extract_topics=True)

    output_data_file = open(os.path.join(output_dir, f"{base_output_name}_topics_indexing.json"), "w+")

    for entry in data_with_topics:
        abstract_text, topics, expected_entities = entry

        topics_text = separator.join(topics)
        expected_entities_text = separator.join(f"\"{item}\"" for item in expected_entities if item != "")

        if topics_with_abstract:
            text = f"<|endoftext|> {prompt_start[1]} {abstract_text} entities: {expected_entities_text}. {prompt_end[1]} {topics_text} <|endoftext|>"
        else:
            text = f"<|endoftext|> {prompt_start[1]} {expected_entities_text}. {prompt_end} <|endoftext|>"

        text = re.sub(" {2,}", " ", text)
        data_entry = {"text": text, "abstract_text": abstract_text, "expected_entities": expected_entities_text}

        json.dump(data_entry, output_data_file)
        output_data_file.write("\n")

    output_data_file.close()


def load_and_write_both(input_path: str, output_dir: str, base_output_name: str, prompt_start: List[str],
                        prompt_end: List[str], separator: str, topics_with_abstract: bool) -> None:
    data_corpus = parse_data(input_path)
    data_with_topics = get_targets(data_corpus, extract_topics=True)

    output_data_file = open(os.path.join(output_dir, f"{base_output_name}_joint_finetune.json"), "w+")

    for entry in data_with_topics:
        abstract_text, topics, expected_entities = entry

        topics_text = separator.join(topics)
        expected_entities_text = separator.join(f"{item}" for item in expected_entities if item != "")

        find_entities_text = f"<|endoftext|> {prompt_start[0]} {abstract_text} {prompt_end[0]} {expected_entities_text} <|endoftext|>"

        if topics_with_abstract:
            find_topics_text = f"<|endoftext|> {prompt_start[1]} {abstract_text} entities: {expected_entities_text}. {prompt_end[1]} {topics_text} <|endoftext|>"
        else:
            find_topics_text = f"<|endoftext|> {prompt_start[1]} {expected_entities_text}. {prompt_end} <|endoftext|>"

        find_entities_text = re.sub(" {2,}", " ", find_entities_text)
        find_topics_text = re.sub(" {2,}", " ", find_topics_text)

        entities_entry = {"text": find_entities_text}
        topics_entry = {"text": find_topics_text, "abstract_text": abstract_text,
                        "expected_entities": expected_entities_text}

        json.dump(entities_entry, output_data_file)
        output_data_file.write("\n")
        json.dump(topics_entry, output_data_file)
        output_data_file.write("\n")

    output_data_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_file", help="Path(s) to JSON files holding the raw data.", type=str, nargs="+",
                        required=True)
    parser.add_argument("--output_dir", help="Path to write the processed dataset(s) to.", type=str, required=True)
    parser.add_argument("--prompt_start", help="String to add to the beginning of the text for prompting purposes.",
                        type=str, required=True, nargs="+")
    parser.add_argument("--prompt_end", help="String to add to the end of the text for prompting purposes", type=str,
                        required=True, nargs="+")
    parser.add_argument("--name_suffix", help="Suffix to add to base name for each file.", type=str, default="")
    parser.add_argument("--mode", help="One of entities, topics or both indicating the whether the dataset "
                                       "should be assembled for NER, topics or combined finetuning. The mode "
                                       "'both' is to produce data for finetuning a model that performs both "
                                       "NER and topics inference. In this case, the prompt_start and prompt_end "
                                       "arguments should have 2 strings each.", type=str,
                        choices=["entities", "topics", "both"], default="entities")
    parser.add_argument("--separator", help="String to separate entities/topics", type=str, default="; ")
    parser.add_argument("--topics_with_abstract", action="store_true",
                        help="Whether or not to incorporate the abstract during data construction for topic inference.")

    args = parser.parse_args()

    for input_file in args.source_file:
        last_slash = input_file.rfind("/")

        if last_slash == -1:
            filename_leaf = input_file

        else:
            filename_leaf = input_file[last_slash + 1:]

        base_name = filename_leaf[:filename_leaf.find(".")]

        if args.name_suffix:
            base_name += f"_{args.name_suffix}"

        if args.mode == "topics":
            load_and_write_topics(input_file, args.output_dir, base_name, args.prompt_start[0], args.prompt_end[0], args.separator, args.topics_with_abstract)
        elif args.mode == "entities":
            load_and_write_data(input_file, args.output_dir, base_name, args.prompt_start[0], args.prompt_end[0], args.separator)
        elif args.mode == "both":
            load_and_write_both(input_file, args.output_dir, base_name, args.prompt_start, args.prompt_end, args.separator, args.topics_with_abstract)
        else:
            raise ValueError(f"Mode should be one of topics, entities or both. Got: {args.mode}.")
