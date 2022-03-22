# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Some code of this file was adapted from the HuggingFace library available at
# https://github.com/huggingface/transformers

import mmap
import itertools
from copy import deepcopy
from typing import Dict, Optional, List, Iterator, TypeVar, Callable

import torch
from torch.utils.data import IterableDataset
from apex.transformer import parallel_state
import numpy as np
from datasets import load_dataset, set_caching_enabled, interleave_datasets
from datasets.iterable_dataset import (
    RandomlyCyclingMultiSourcesExamplesIterable,
    MappedExamplesIterable,
    BufferShuffledExamplesIterable,
    _BaseExamplesIterable
)
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.t0_task_manager import (
    DATA_ORG, t0_all_evaldt_names_subset,
    get_data_paths_and_splits
)
from nemo.collections.nlp.data.language_modeling.t0_task_manager import (
    get_task_id,
    TEMPLATE_CHUNK_NAME,
    ORIG_TXT_CHUNK_NAME
)
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import (
    make_attention_mask_3d,
    make_history_mask_3d,
)
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

T_co = TypeVar('T_co', covariant=True)


class Task(object):
    def __init__(self, file_path: str,  dt_name: str, subset: str):
        self.file_path = file_path
        self.task_id = get_task_id(dt_name, subset)
        self.prompt_id = {}
        self.dataset_size = self.mapcount()

    def mapcount(self):
        f = open(self.file_path, "r+")
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
        return lines

    def map_fn(self, multi_prompted_ex):
        features = []
        for prompt_type, data in multi_prompted_ex.items():
            if data is None:
                continue
            self.prompt_id[prompt_type] = self.prompt_id.get(prompt_type, len(self.prompt_id) + 1)
            example = self.create_example(data, self.task_id, self.prompt_id[prompt_type])
            feature_dicts = self.tokenize(example)
            features.append(feature_dicts)
        return features


class InputPromptedExample(object):
    """A single training/test example for prompted inputs.

    Args:
        task_id: Unique id for the example.
        text: The untokenized text of the first sequence.
        For single sequence tasks, only this sequence must be specified.
        prompt_type: Name of prompt applied to the exampled.
        label:The label of the example. This should be
        specified for train and dev examples, but not for test examples.
        chunked_idx: a prompt is already applied on original text,
        we can recover the template text using the idx
    """

    def __init__(self, task_id: int, text: str, prompt_id: int = None, label: str = None, chunked_idx: List[List[int]] = None):
        """Constructs a InputExample."""
        self.task_id = task_id
        self.input_text = text
        self.prompt_id = prompt_id
        self.label = label
        self.chunked_idx =chunked_idx

    def __repr__(self):
        return (
            f"InputExample(task_id='{self.task_id}', input_text='{self.input_text}', "
            f"prompt_type='{self.prompt_id}', label='{self.label}'), "
            f"template_idx='{self.template_idx}', original_text_idx={self.original_text_idx}."
        )


class T0DatasetBuilder(object):
    """T0 Dataset Builder in a text-to-text format."""

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return

    def __init__(
            self,
            t0_type: str,
            dir_path: str,
            max_sampling_size: int,
            split: str,
            tokenizer: TokenizerSpec,
            max_seq_length: int,
            max_seq_length_decoder: int = 128,
            seed: int = 43,
            buffer_size: int = 10_000,
            chunk_size: int = 40 << 20,
            use_cache: bool = True,
            extension: str = 'json',
            max_samples: int = None
    ):
        """
        Processes T0 dataset
        Args:
            t0_type: [train_t0, train_t0p, train_t0pp, t0_debug]
            split: train, test, eval
            max_sampling_size: caps probabilities at max size for a specific dataset
            dir_path: path to data directory for train, evaluation or test
            tokenizer: such as AutoTokenizer
            max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
            max_seq_length_decoder: max sequence length
            buffer_size: size of the buffer, chunks of data to suffle
            use_cache: whether to use data cache
            max_samples: limit size of dataset (not implemented)
        """
        self.t0_type = t0_type
        self.dir_path = dir_path
        self.max_sampling_size = max_sampling_size
        self.split = split
        self.tokenizer = tokenizer
        self.max_query_length = max_seq_length
        self.max_query_length_decoder = max_seq_length_decoder
        self.seed = seed
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.extension = extension
        self.max_samples = max_samples
        self.tasks = []
        set_caching_enabled(use_cache)
        self.datasets = self.get_data_dict()

    def update_ex_iterable(self, dataset):
        """This hacks MappedExamplesIterable and RandomlyCyclingMultiSourcesExamplesIterable"""
        if isinstance(dataset, dict):
            for key, torch_iter in dataset.items():
                mapped_itr = torch_iter._ex_iterable
                update_itr = MyMappedExamplesIterable(
                    ex_iterable=mapped_itr.ex_iterable,
                    function=mapped_itr.function,
                    batched=mapped_itr.batched,
                    batch_size=mapped_itr.batch_size
                )
                dataset[key]._ex_iterable = update_itr
        else:
            rand_multisrc_iter = dataset._ex_iterable
            assert isinstance(rand_multisrc_iter, RandomlyCyclingMultiSourcesExamplesIterable)
            updated_buffered_itr = []
            for buffered_itr in rand_multisrc_iter.ex_iterables:
                assert isinstance(buffered_itr, BufferShuffledExamplesIterable)
                mapped_itr = buffered_itr.ex_iterable
                update_itr = MyMappedExamplesIterable(
                    ex_iterable=mapped_itr.ex_iterable,
                    function=mapped_itr.function,
                    batched=mapped_itr.batched,
                    batch_size=mapped_itr.batch_size,
                    output_function=True
                )
                buffered_itr.ex_iterable = update_itr
                updated_buffered_itr.append(buffered_itr)
            dataset._ex_iterable = MyRandomlyCyclingMultiSourcesExamplesIterable(
                ex_iterables=updated_buffered_itr,
                generator=rand_multisrc_iter.generator,
                probabilities=rand_multisrc_iter.probabilities
            )
        return dataset

    def assemble_datasets(self):
        if self.split == 'train':
            datasets_list = list(self.datasets.values())
            datasets_list = [d.shuffle(buffer_size=self.buffer_size, seed=self.seed) for d in datasets_list]
            datasets = interleave_datasets(
                datasets=datasets_list,
                probabilities=self.get_sampling_probs(),
                seed=self.seed
            )
            datasets.info.dataset_size = len(self)
            datasets = self.update_ex_iterable(datasets)
            datasets = datasets.with_format('torch')
            return datasets
        else:
            return self.update_ex_iterable(self.datasets)

    def get_dataset(self, task):
        dataset = load_dataset(
            self.extension, data_files=task.file_path, streaming=True, chunksize=self.chunk_size
        )
        dataset = dataset['train'].map(task.map_fn, batched=False)
        dataset = dataset.with_format('torch')
        dataset.info.dataset_size = task.dataset_size
        dataset.task = task
        return dataset

    def get_task(self, file_path, dt_name, subset):
        task = Task(file_path, dt_name, subset)
        task.create_example = self.create_example
        task.tokenize = self.tokenize
        self.tasks.append(task)
        return task

    def get_data_dict(self):
        if self.split == 'train':
            data_dict = DATA_ORG[self.t0_type]
        else:
            data_dict = t0_all_evaldt_names_subset
        dataset_dict = {}
        for dt_name in data_dict.keys():
            logging.info('Dataset name %s.' % dt_name)
            subsets = data_dict[dt_name]
            if not isinstance(subsets, list):
                subsets = [subsets]
            for subset in subsets:
                logging.info('Subset name %s.' % subset)
                if "/" in dt_name:
                    dt_name = dt_name.split("/")[-1]
                file_name = "_%s_%s.jsonl" % (dt_name, "" if subset is None else subset)
                _, data_paths = get_data_paths_and_splits(self.split, self.dir_path, file_name, dt_name)
                for file_path in data_paths:
                    task = self.get_task(file_path, dt_name, subset)
                    task_name = "%s_%s" % (dt_name, "" if subset is None else subset)
                    dataset_dict[task_name] = self.get_dataset(task)
        return dataset_dict

    def get_sampling_probs(self):
        sampling_data_sizes = []
        for dataset in self.datasets.values():
            sampling_data_sizes.append(min(dataset.dataset_size, self.max_sampling_size))
        sampling_data_sizes = np.array(sampling_data_sizes)
        sampling_probs = sampling_data_sizes / np.sum(sampling_data_sizes)
        return sampling_probs.tolist()

    def __len__(self):
        return sum(d.dataset_size for d in self.datasets.values())

    @staticmethod
    def choose_template(features):
        num_prompts = len(features)
        choose_prompt = np.random.randint(0, num_prompts)
        return features[choose_prompt]

    @staticmethod
    def create_example(data, task_id, prompt_id):
        return InputPromptedExample(
            task_id=task_id,
            text=data['input'],
            prompt_id=prompt_id,
            label=data['output']
        )

    def collate_fn(self, batch):
        new_batch = []
        for features in batch:
            feature = self.choose_template(features)
            new_batch.append(feature)
        return self.collate_fn2(new_batch)

    def tokenize(self, example):
        enc_query = self.tokenizer.text_to_ids(example.input_text)
        if len(enc_query) > self.max_query_length:
            enc_query = enc_query[: self.max_query_length]
        dec_query = (
                [self.tokenizer.cls_id]
                + self.tokenizer.text_to_ids(example.label)
                + [self.tokenizer.eos_id]
        )
        if len(dec_query) > self.max_query_length_decoder + 1:
            dec_query = dec_query[: self.max_query_length_decoder + 1]
        dec_input = dec_query[:-1]
        labels = dec_query[1:]
        task_id = [example.task_id]
        prompt_id = [example.prompt_id]
        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'task_id': task_id,
            'prompt_id': prompt_id
        }

    def collate_fn2(self, batch):
        enc_query = [item['text_enc'] for item in batch]
        dec_input = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]
        task_ids = [item['task_id'] for item in batch]
        prompt_ids = [item['prompt_id'] for item in batch]

        max_dec_input_length = max([len(item) for item in dec_input])
        max_enc_query_length = max([len(item) for item in enc_query])
        max_label_length = max([len(item) for item in labels])

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_query = [item + [self.tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        dec_input = [item + [self.tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
        labels = [item + [self.tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]

        enc_query = torch.LongTensor(enc_query)
        dec_input = torch.LongTensor(dec_input)
        labels = torch.LongTensor(labels)
        task_ids = torch.LongTensor(task_ids)
        prompt_ids = torch.LongTensor(prompt_ids)
        loss_mask = torch.LongTensor(loss_mask)

        enc_mask = make_attention_mask_3d(enc_query, enc_query, self.tokenizer.pad_id).long()
        dec_mask = make_attention_mask_3d(dec_input, dec_input, self.tokenizer.pad_id)
        dec_mask = (dec_mask * make_history_mask_3d(dec_input)).long()
        enc_dec_mask = make_attention_mask_3d(dec_input, enc_query, self.tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'enc_dec_mask': enc_dec_mask,
            'task_ids': task_ids,
            'prompt_ids': prompt_ids
        }

    def make_history_mask_3d(self, block):
        batch, length = block.shape
        arange = np.arange(length)
        history_mask = (arange[None,] <= arange[:, None])[
            None,
        ]
        history_mask = np.repeat(history_mask, batch, 0)
        return history_mask


class T0PrimeDatasetBuilder(T0DatasetBuilder):
    """T0' Dataset Builder in a text-to-text format."""

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return

    def __init__(
            self,
            t0_type: str,
            dir_path: str,
            max_sampling_size: int,
            split: str,
            tokenizer: TokenizerSpec,
            max_seq_length: int,
            prompt_token_id: int,
            prompt_seq_len: int,
            max_seq_length_decoder: int = 128,
            seed: int = 43,
            buffer_size: int = 10_000,
            chunk_size: int = 40 << 20,
            use_cache: bool = True,
            extension: str = 'json',
            max_samples: int = None,
            split_template: bool = True,
    ):
        """
        Processes T0' dataset using differentiable prompts
        Args:
            t0_type: [train_t0, train_t0p, train_t0pp, t0_debug]
            split: train, test, eval
            max_sampling_size: caps probabilities at max size for a specific dataset
            dir_path: path to data directory for train, evaluation or test
            tokenizer: such as AutoTokenizer
            max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
            max_seq_length_decoder: max sequence length
            buffer_size: size of the buffer, chunks of data to suffle
            use_cache: whether to use data cache
            max_samples: limit size of dataset (not implemented)
            prompt_token_id: tokenizer id for [PROMPT] token
            prompt_seq_len: sequence lenght of prompt
            split_template: whether to seperate template tokens
        """
        self.prompt_token_id = prompt_token_id
        self.prompt_seq_len = prompt_seq_len
        self.split_template = split_template
        super().__init__(
            t0_type, dir_path, max_sampling_size, split, tokenizer, max_seq_length, max_seq_length_decoder,
            seed, buffer_size, chunk_size, use_cache, extension, max_samples
        )

    @staticmethod
    def create_example(data, task_id, prompt_id):
        return InputPromptedExample(
            task_id=task_id,
            text=data['input'],
            prompt_id=prompt_id,
            label=data['output'],
            chunked_idx=data['chunked_idx']
        )

    def tokenize(self, example):
        def get_text_chunks(input_text, chunked_idx):
            """
            Splits in the input text into chunks such that:
            chunked(input_text) = [[orig_txt_part1], [template_part1], [orig_txt_part2], [template_part2], ...]
            or other interleaved patterns of template and original text.
            Assumes already sorted lists of index ranges.
            """
            text_chunks = []
            for chunk in chunked_idx.split(","):
                chunk_name, chunk_start, chunk_end = chunk.split("-")
                text_chunks.append((chunk_name, input_text[int(chunk_start):int(chunk_end)]))
            return text_chunks

        input_text_chunks = get_text_chunks(example.input_text, example.chunked_idx)
        enc_query = []
        template = []
        for chunk in input_text_chunks:
            chunk_name = chunk[0]
            chunk_tokens = self.tokenizer.text_to_ids(chunk[1])
            if chunk_name == TEMPLATE_CHUNK_NAME and self.split_template:
                remain = max(0, self.prompt_seq_len - len(template) - len(chunk_tokens))
                template.extend(chunk_tokens[:remain])
                enc_query.extend([self.prompt_token_id] * len(chunk_tokens[:remain]))
            else:
                max_length = self.max_query_length + (0 if self.split_template else self.prompt_seq_len)
                remain = max(0, max_length - len(enc_query) - len(chunk_tokens))
                enc_query.extend(chunk_tokens[:remain])  # only reduce original chunk
        dec_query = (
                [self.tokenizer.cls_id]
                + self.tokenizer.text_to_ids(example.label)
                + [self.tokenizer.eos_id]
        )
        if len(dec_query) > self.max_query_length_decoder + 1:
            dec_query = dec_query[: self.max_query_length_decoder + 1]
        dec_input = dec_query[:-1]
        labels = dec_query[1:]
        task_id = [example.task_id]
        prompt_id = [example.prompt_id]
        return {
            'text_enc': enc_query,
            'template': template,
            'text_dec': dec_input,
            'labels': labels,
            'task_id': task_id,
            'prompt_id': prompt_id
        }

    def collate_fn2(self, batch):
        enc_query = [item['text_enc'] for item in batch]
        template = [item['template'] for item in batch]
        dec_input = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]
        task_ids = [item['task_id'] for item in batch]
        prompt_ids = [item['prompt_id'] for item in batch]
        
        if self.split_template:
            max_template_length = max(self.prompt_seq_len, max([len(item) for item in template]))
            enc_query = [item_q + [self.prompt_token_id] * (max_template_length - len(item_t)) for item_q, item_t in zip(enc_query, template)]

        max_dec_input_length = max([len(item) for item in dec_input])
        max_enc_query_length = max([len(item) for item in enc_query])
        max_label_length = max([len(item) for item in labels])

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_query = [item + [self.tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        template = [item[:self.prompt_seq_len] + [self.tokenizer.pad_id] * (self.prompt_seq_len - len(item)) for item in template]
        dec_input = [item + [self.tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
        labels = [item + [self.tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]

        enc_query = torch.LongTensor(enc_query)
        template = torch.LongTensor(template)
        dec_input = torch.LongTensor(dec_input)
        labels = torch.LongTensor(labels)
        task_ids = torch.LongTensor(task_ids)
        prompt_ids = torch.LongTensor(prompt_ids)
        loss_mask = torch.LongTensor(loss_mask)

        enc_mask = make_attention_mask_3d(enc_query, enc_query, self.tokenizer.pad_id).long()
        dec_mask = make_attention_mask_3d(dec_input, dec_input, self.tokenizer.pad_id)
        dec_mask = (dec_mask * make_history_mask_3d(dec_input)).long()
        enc_dec_mask = make_attention_mask_3d(dec_input, enc_query, self.tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'template': template,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'enc_dec_mask': enc_dec_mask,
            'task_ids': task_ids,
            'prompt_ids': prompt_ids
        }


class MyRandomlyCyclingMultiSourcesExamplesIterable(RandomlyCyclingMultiSourcesExamplesIterable):
    """Same as RandomlyCyclingMultiSourcesExamplesIterable but iter accounts for multiple workers/GPUs."""
    def __init__(self, ex_iterables, generator: np.random.Generator, probabilities: Optional[List[float]] = None):
        super().__init__(ex_iterables, generator, probabilities)
        self.rank = parallel_state.get_data_parallel_rank()
        self.world_size = parallel_state.get_data_parallel_world_size()

    def __iter__(self):
        #TODO include model parallel case
        rng = deepcopy(self.generator)
        worker_info = torch.utils.data.get_worker_info()
        total_workers_per_process = 1 if worker_info is None else worker_info.num_workers
        worker_id = (0 if worker_info is None else worker_info.id) + total_workers_per_process * self.rank
        total_workers = total_workers_per_process * self.world_size
        iterators = [iter(ex_iterable) for ex_iterable in self.ex_iterables]
        iterators = [itertools.islice(itr, worker_id, None, total_workers) for itr in iterators]
        iterators = [itertools.tee(itr) for itr in iterators]
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        indices_iterator = self._iter_random_indices(rng, len(iterators), p=self.probabilities)
        expected_itr_idx = [i for i, p in enumerate(self.probabilities) if 0 < p < 1]
        finished_itr_idx = []
        for i in indices_iterator:
            try:  # let's pick one example from the iterator at index i
                key, example, func = next(iterators[i][0])
                yield key, func(example)
            except StopIteration:  # if we ran out of examples on this iterator, break the main for loop
                if i not in finished_itr_idx:
                    finished_itr_idx.append(i)
                    finished_itr_idx.sort()
                if finished_itr_idx == expected_itr_idx:
                    break
                else:
                    iterators[i] = itertools.tee(iterators[i][1])

    def shuffle_data_sources(self, generator: np.random.Generator) -> "MyRandomlyCyclingMultiSourcesExamplesIterable":
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return MyRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables, generator=generator, probabilities=self.probabilities
        )


class MyMappedExamplesIterable(MappedExamplesIterable):
    def __init__(
            self,
            ex_iterable: _BaseExamplesIterable,
            function: Callable,
            with_indices: bool = False,
            input_columns: Optional[List[str]] = None,
            batched: bool = False,
            batch_size: int = 1000,
            remove_columns: Optional[List[str]] = None,
            output_function: Optional[bool] = False
    ):
        super().__init__(
            ex_iterable, function, with_indices, input_columns, batched, batch_size, remove_columns
        )
        self.output_function = output_function

    def __iter__(self):
        assert not self.batched, "Current implementation does handle batched examples. " \
                               "Use hugginface's dataset.MappedExamplesIterable instead."
        iterator = iter(self.ex_iterable)
        for key, example in iterator:
            # If not batched, apply the transform and yield the example directly
            if self.output_function:
                yield key, example, self.function
            else:
                yield key, self.function(example)

    def shuffle_data_sources(self, seed: Optional[int]) -> "MyMappedExamplesIterable":
        """Shuffle the wrapped examples iterable."""
        return MyMappedExamplesIterable(
            self.ex_iterable.shuffle_data_sources(seed),
            function=self.function,
            batched=self.batched,
            batch_size=self.batch_size,
        )

