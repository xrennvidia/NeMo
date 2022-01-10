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

import torch
import json
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader
import pickle

from nemo.collections.nlp.data.language_modeling.megatron.gpt_request_dataset import GPTRequestDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from nemo.utils.app_state import AppState

assert torch.cuda.is_available()


def parse_inferred_entities(inferred_entities: List[str]) -> List[str]:
    """
    Parse the inferred entities originally stored for comparisons between the prompt, inferred entities and expected
    entities to extract only the inferred entities.

    The ordering of the entities are assumed
    """
    parsed_entities = []

    for line in inferred_entities:
        if line.startswith("Completed prompt"):
            line = line.strip()
            entities_start = line.find("answers: ") + len("answers: ")
            entities = line[entities_start:]

            parsed_entities.append(entities)

    return parsed_entities


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default="", required=True, help="Pass path to model's .nemo file")
    parser.add_argument(
        "--path_to_file", type=str, required=True, help="Path to file with prompts (a text to complete)"
    )
    parser.add_argument(
        "--path_to_entities", type=str, required=True, help="Path to file with the inferred entities"
    )
    parser.add_argument(
        "--tokens_to_generate", type=int, default="64", required=False, help="How many tokens to add to prompt"
    )
    parser.add_argument(
        "--stop_after_sentence",
        type=bool,
        default="True",
        required=False,
        help="True/False: whether to stop after full sentence has been generated.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to directory to output predictions."
    )
    parser.add_argument(
        "--tensor_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument("--precision", default=16, help="PyTorch Lightning Trainer precision flag")

    args = parser.parse_args()

    # cast precision to int if 32 or 16
    if args.precision in ["32", "16"]:
        args.precision = int(float(args.precision))

    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=NLPDDPPlugin(), gpus=args.tensor_model_parallel_size, precision=args.precision)

    app_state = AppState()
    if args.tensor_model_parallel_size is not None and args.tensor_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size
        app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

    model = MegatronGPTModel.restore_from(restore_path=args.model_file, trainer=trainer)

    model.freeze()

    # defining type of request
    with open(args.path_to_file) as f:
        data = f.readlines()

    with open(args.path_to_entities) as f:
        inferred_entities = f.readlines()

    parsed_entities = parse_inferred_entities(inferred_entities)

    original_prompts = []
    prompts = []
    expected_entities = []
    labels = []
    for line, entities in zip(data, parsed_entities):
        line = json.loads(line)
        text = line["text"]
        prompt_end = text.find("Answers:") + 8

        if prompt_end < 8:
            raise ValueError("\"answers\" not found in text.")

        prompts.append(f"<|endoftext|> Find topics: {entities}. Answers:")
        expected_entities.append(line["expected_entities"])
        original_prompts.append(line["abstract_text"])
        labels.append(text[prompt_end + 1: text.rfind("<|endoftext|>")])

    all_data = []
    for prompt in prompts:
        request = {
            "prompt": prompt,
            "tokens_to_generate": args.tokens_to_generate,
            "stop_after_sentence": args.stop_after_sentence
        }
        all_data.append(request)

    dataset = GPTRequestDataset(all_data, model.tokenizer)
    request_dl = DataLoader(dataset)
    response = trainer.predict(model, request_dl)

<<<<<<< HEAD
=======
    pickle.dump(response, open("/nlp_project/topics_response.txt", "wb"))

>>>>>>> Updated data processing with code to prepare topic indexing data; Added code to perform inference; Added code to measure metrics for entities inference
    with open(args.output_file, "w+") as f:
        for original_prompt, response_item, label in zip(original_prompts, response[0], labels):
            prompt = response_item["prompt"]
            completion = response_item["completion"]["text"].strip()
            print(f"Original prompt:  {original_prompt}\n\n")
            print(f"Pred entities:    {prompt[0].strip('[]')}\n\n")
            print(f"Pred topics:      {completion}\n\n")
            print(f"Expected topics:  {label}\n\n")
            print("*********************************************************\n\n")
            f.write(f"Original prompt:  {original_prompt}\n\n")
            f.write(f"Pred entities:    {prompt[0].strip('[]')}\n\n")
            f.write(f"Pred topics:      {completion}\n\n")
            f.write(f"Expected topics:  {label}\n\n")
            f.write("*********************************************************\n\n")


if __name__ == '__main__':
<<<<<<< HEAD
    main()  # noqa pylint: disable=no-value-for-parameter
=======
    main()  # noqa pylint: disable=no-value-for-parameter
>>>>>>> Updated data processing with code to prepare topic indexing data; Added code to perform inference; Added code to measure metrics for entities inference
