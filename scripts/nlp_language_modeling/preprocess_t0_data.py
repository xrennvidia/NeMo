# coding=utf-8
import os
import re
import sys
import nltk
import json
from datasets import load_dataset
from argparse import ArgumentParser
from nltk.tokenize import sent_tokenize
from promptsource.templates import DatasetTemplates
sys.path.append("../../")
from nemo.collections.nlp.data.language_modeling.t0_task_manager import (
    get_data_paths_and_splits,
    t0pp_traindt_names_subset,
    t0_all_evaldt_names_subset,
    t0_debug, TEMPLATE_CHUNK_NAME, ORIG_TXT_CHUNK_NAME
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

special_dt_dir ={
    "story_cloze": "/home/jpilault/datasets/downloads/story_cloze_2016/"
}
MAX_CHARS = 1500

def get_text_template_idx(example, templated_text):
    elements = []
    for key, val in example.items():
        if key == 'concepts':
            # for common_gen
            el = val
        elif key == 'choices' and isinstance(val, dict)  and 'text' in val:
            # for qasc
            el = val['text']
        elif isinstance(val, dict) and 'table' in val:
            # for wiki_bio
            el = val['table']['column_header'] + [e.strip() for e in val['table']['content']]
        elif not isinstance(val, str):
            continue
        elif val.find('@highlight') != -1:
            # for super_glue/record
            el = val.split('@highlight')
        elif len(val) < MAX_CHARS:
            el = [val]
        else:
            # for long inputs that are clipped in template (race, cnn-dm, imdb, ...)
            el = []
            tok_el = sent_tokenize(val)
            for sent in tok_el:
                if len(sent) > MAX_CHARS:
                    el.extend(re.split(r'\\.\\.\\.|,|[.]', sent))
                else:
                    el.append(sent)
        elements.extend(el)
    elements.sort(key=len)
    elements.reverse()

    i = len(elements) - 1
    while i > -1:
        if not elements[i].lower().strip() in templated_text.lower():
            elements.pop(i)
        i -= 1

    i = len(elements) - 1
    while i > 0:
        if elements[i].lower().strip() in elements[0].lower().strip():
            elements.pop(i)
        i -= 1

    original_text_idx = []
    for el in elements:
        el = el.replace("\'", r"'")
        idx = templated_text.find(el)
        if idx == -1:
            el = el.lower().strip()
            idx = templated_text.lower().find(el)
        if idx != -1:
            idx_range = [idx, idx + len(el)]
            if idx_range not in original_text_idx:
                original_text_idx.append(idx_range)
    original_text_idx.sort()

    pointer = 0
    original_text_chunk_idx = 0
    template_idx = []
    while pointer < len(templated_text):
        txt_start, txt_end = original_text_idx[original_text_chunk_idx]
        if txt_start > pointer:
            template_idx.append([pointer, txt_start])
        pointer = txt_end
        original_text_chunk_idx += 1
        if original_text_chunk_idx > len(original_text_idx) - 1:
            if pointer < len(templated_text):
                template_idx.append([pointer, len(templated_text)])
                pointer = len(templated_text)
    template_idx.sort()
    return original_text_idx, template_idx

def organize_chunk_idx(original_text_idx, template_idx):
    tuple_list = []
    while len(template_idx) > 0 or len(original_text_idx) > 0:
        if not original_text_idx:
            tuple_list.append((TEMPLATE_CHUNK_NAME, template_idx.pop(0)))
        elif not template_idx:
            tuple_list.append((ORIG_TXT_CHUNK_NAME, original_text_idx.pop(0)))
        elif template_idx[0] < original_text_idx[0]:
            tuple_list.append((TEMPLATE_CHUNK_NAME, template_idx.pop(0)))
        else:
            tuple_list.append((ORIG_TXT_CHUNK_NAME, original_text_idx.pop(0)))
    return tuple_list

def apply_prompts(dataset, prompts, splits, save_paths):
    for split, save_path in zip(splits, save_paths):
        counter = 0
        printed = False
        with open(save_path, 'w') as f:
            for example in dataset[split]:
                row = {}
                for template_name in prompts.name_to_id_mapping:
                    prompt = prompts[template_name]
                    if not prompt.metadata.original_task:
                        continue
                    result = prompt.apply(example)
                    if not result[0]:
                        continue
                    try:
                        templated_text = result[0]
                        output = result[1]
                        original_text_idx, template_idx = get_text_template_idx(example, templated_text)
                        chunked_idx = organize_chunk_idx(original_text_idx, template_idx)
                        assert chunked_idx[0][1][0] == 0
                        #assert any(c[1][1] == len(templated_text) for c in chunked_idx)
                        row[template_name] = {
                            'input': templated_text, 'output': output, 'chunked_idx': chunked_idx
                        }
                    except IndexError:
                        if not printed:
                            print("ISSUE DETECTED")
                            #original_text_idx, template_idx = get_text_template_idx(example, templated_text)
                            print(save_path)
                            print(template_name)
                        printed = True
                        continue

                f.write(json.dumps(row))
                f.write('\n')
                counter += 1
                if counter % 100000 == 0:
                    print("{counter} applied...".format(counter=counter))
        print(f"Saved {counter} examples...")

def save_raw_jsonl(dataset, prompts, splits, save_paths):
    for split, save_path in zip(splits, save_paths):
        counter = 0
        with open(save_path, 'w') as f:
            for example in dataset[split]:
                f.write(json.dumps(example))
                f.write('\n')
                counter += 1
                if counter % 100000 == 0:
                    print("{counter} applied...".format(counter=counter))

def preprocess_data(data_dict, main_splits, data_dir, save_raw):
    for dt_name in data_dict.keys():
        print(dt_name)
        subsets = data_dict[dt_name]
        if not isinstance(subsets, list):
            subsets = [subsets]
        for subset in subsets:
            print(subset)
            dataset = load_dataset(dt_name, subset, data_dir=special_dt_dir.get(dt_name, None))
            prompts = DatasetTemplates(dt_name, subset)
            file_name = "_%s_%s.jsonl" % (dt_name, "" if subset is None else subset)
            splits, save_paths = get_data_paths_and_splits(main_splits, data_dir, file_name, dt_name)
            if save_raw:
                save_raw_jsonl(dataset, prompts, splits, save_paths)
            else:
                apply_prompts(dataset, prompts, splits, save_paths)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_split", type=str, choices=['train', 'test', 'debug'], help="Dataset split you want to prepare ['train', 'test']."
    )
    parser.add_argument(
        "--data_dir", type=str, default="/home/jpilault/datasets/T0_prompted", help="Parent t0 directory."
    )
    parser.add_argument(
        "--save_raw", action='store_true', default=False, help="Just save raw files to disk."
    )
    args = parser.parse_args()


    load_dataset("story_cloze", "2016", data_dir="/home/jpilault/datasets/downloads/story_cloze_2016/")

    if args.dataset_split == "train":
        data_dict = t0pp_traindt_names_subset
        main_splits = ['train']
    elif args.dataset_split == "debug":
        data_dict = t0_debug
        main_splits = ['train']
    else:
        data_dict = t0_all_evaldt_names_subset
        main_splits = ['test', 'validation']

    preprocess_data(data_dict, main_splits, args.data_dir, args.save_raw)


if __name__ == '__main__':
    main()