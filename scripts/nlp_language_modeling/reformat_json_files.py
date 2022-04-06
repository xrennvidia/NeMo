import glob
import json
from tqdm import tqdm
from copy import copy
from datasets import load_dataset


def create_test_json(dir, num_subsets=10):
    prefix = dir + "train_dummy_dataset_"
    num_examples = 1000
    start = 0
    for i in range(num_subsets):
        end = start + num_examples
        with open(prefix + str(i) + ".jsonl", "w") as f:
            for j in range(start, end):
                f.write(json.dumps({"a": [j]}))
                f.write('\n')
        start = end
        num_examples -= 90
    print(end)


def reformat_json(file_list):
    for file in tqdm(file_list):
        name = file.split("/")[-1]
        new_file = new_dir + name
        with open(file, "r") as fr, open(new_file, "w") as fw:
            for line in fr:
                json_line = json.loads(line)
                for template_names in json_line.keys():
                    chunks = json_line[template_names]['chunked_idx']
                    new_chunks = ''
                    for chunk in chunks:
                        if new_chunks:
                            new_chunks += ', '
                        new_chunks += chunk[0] + '-' + str(chunk[1][0]) + '-' + str(chunk[1][1])
                    json_line[template_names]['chunked_idx'] = new_chunks
                fw.write(json.dumps(json_line))
                fw.write('\n')


def remove_none(file_list):
    # removes template names that have None values
    for file in tqdm(file_list):
        name = file.split("/")[-1]
        new_file = new_dir + name
        with open(file, "r") as fr, open(new_file, "w") as fw:
            for line in fr:
                json_line = json.loads(line)
                new_jsonline = {}
                for template_names in json_line.keys():
                    if json_line[template_names] is None:
                        continue
                    else:
                        new_jsonline[template_names] = json_line[template_names]
                if new_jsonline:
                    fw.write(json.dumps(new_jsonline))
                    fw.write('\n')


def add_empty_fields(file_list):
    # reverses remove_none() by adding template names with None as values
    for file in tqdm(file_list):
        name = file.split("/")[-1]
        new_file = new_dir + name
        data = []
        prompt_names = set()
        with open(file, "r") as fr:
            for line in fr:
                json_line = json.loads(line)
                data.append(json_line)
                prompt_names.update(set(json_line.keys()))

        with open(new_file, "w") as fw:
            base_json_format = {k: {'input': None, 'output': None, 'chunked_idx': None} for k in list(prompt_names)}
            for dt in data:
                new_jsonline = copy(base_json_format)
                new_jsonline.update(dt)
                fw.write(json.dumps(new_jsonline))
                fw.write('\n')


def check_outputs(file_list):
    for file in tqdm(file_list):
        with open(file, "r") as fr:
            for line in fr:
                outputs = []
                json_line = json.loads(line)
                for template_names in json_line.keys():
                    outputs.append(json_line[template_names]['output'])
                assert len(set(outputs)) < 3


def make_single_line(file_list):
    for file in tqdm(file_list):
        name = file.split("/")[-1]
        new_file = new_dir + name
        with open(file, "r") as fr, open(new_file, "w") as fw:
            for line in fr:
                json_line = json.loads(line)
                for template_names in json_line.keys():
                    new_jsonline = {}
                    if json_line[template_names] is None:
                        continue
                    else:
                        new_jsonline[template_names] = json_line[template_names]
                    if new_jsonline:
                        fw.write(json.dumps(new_jsonline))
                        fw.write('\n')


if __name__ == '__main__':
    old_dir = "/home/jpilault/datasets/T0_prompted/test_old/"
    new_dir = "/home/jpilault/datasets/T0_prompted/test/"

    file_list = glob.glob(old_dir + "*jsonl")

    make_single_line(file_list)
    #add_empty_fields(file_list)
    #remove_none(file_list)
    #reformat_json(file_list)
    #check_outputs(file_list)
    #create_test_json(new_dir)
