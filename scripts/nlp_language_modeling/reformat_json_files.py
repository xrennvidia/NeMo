import glob
import json
from tqdm import tqdm
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


def check_outputs(file_list):
    for file in tqdm(file_list):
        with open(file, "r") as fr:
            for line in fr:
                outputs = []
                json_line = json.loads(line)
                for template_names in json_line.keys():
                    outputs.append(json_line[template_names]['output'])
                assert len(set(outputs)) < 3


if __name__ == '__main__':
    old_dir = "/home/jpilault/datasets/T0_prompted/train_old/"
    new_dir = "/home/jpilault/datasets/T0_prompted/train/"

    file_list = glob.glob(old_dir + "*")

    reformat_json(file_list)
    #check_outputs(file_list)
    #create_test_json(new_dir)
