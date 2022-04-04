# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import torch
import numpy as np

from nemo.collections.nlp.data.language_modeling.megatron.megatron_dataset import MegatronDataset
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_indexed_dataset_
from nemo.utils import logging

class MegatronNMTDataset(MegatronDataset):
    """Machine Translation Dataset based on Megatron Dataset Utils."""

    def __init__(
        self,
        cfg,
        trainer,
        encoder_tokenizer,
        decoder_tokenizer,
        src_dataset_prefix,
        tgt_dataset_prefix,
        start_index,
        end_index,
        max_encoder_seq_length,
        max_decoder_seq_length,
        data_impl='mmap',
        skip_warmup=True,
        dataset_index_sampling_override=False,
    ):
        super().__init__(cfg, trainer=trainer)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.src_dataset_prefix = src_dataset_prefix
        self.tgt_dataset_prefix = tgt_dataset_prefix
        self.data_impl = data_impl
        self.start_index = start_index
        self.end_index = end_index
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.src_indexed_dataset = self._get_indexed_dataset(src_dataset_prefix, data_impl, skip_warmup)
        self.tgt_indexed_dataset = self._get_indexed_dataset(tgt_dataset_prefix, data_impl, skip_warmup)
        assert len(self.src_indexed_dataset) == len(self.tgt_indexed_dataset)
        self._dataset_length = lambda dataset: dataset.sizes.shape[0]
        if dataset_index_sampling_override:
            logging.info("Disregarding index selections and making up our own.")
            # Make sure the shuffling is different per rank so that the same
            # samples aren't selected by every worker.
            self.rng = np.random.default_rng(seed=torch.distributed.get_rank())

            dataset_len = len(self)
            if dataset_len <= np.iinfo(np.uint32).max:
                dtype = np.uint32
            else:
                dtype = np.uint64
            self.sampling = np.arange(dataset_len, dtype=dtype)
            self.rng.shuffle(self.sampling)

            self.sampling_indx = 0
        else:
            self.sampling = None

    def __len__(self):
        self.end_index - self.start_index

    def _get_indexed_dataset(self, data_prefix, data_impl, skip_warmup):
        indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)
        return indexed_dataset

    def _print_stats(name, start, end):
        logging.info('    {}:'.format(name))
        logging.info('     sentence indices in [{}, {}) total of {} '
                        'sentences'.format(start, end, end - start))

    def build_named_dataset(name, data_prefix, start_index=0, end_index=0):
        src_indexed_dataset, tgt_indexed_dataset = _load_data_prefix(data_prefix)
        if not end_index:
            end_index = self._dataset_length(src_indexed_dataset) - 1
        _print_stats(name, start_index, end_index)
        dataset = Dataset(src_indexed_dataset, tgt_indexed_dataset, start_index,
                            end_index, args.seq_length, args.decoder_seq_length)
        return dataset

    def split_and_build_dataset(split, data_prefix):
        src_indexed_dataset, tgt_indexed_dataset = _load_data_prefix(data_prefix)
        total_num_of_sentences = _dataset_length(src_indexed_dataset) - 1
        splits = get_train_valid_test_split_(splits_string, total_num_of_sentences)
        assert splits[0] < splits[1] < splits[2], \
            "You need to have a train and valid dataset. Splits: {}".format(splits)

        _print_stats('train', splits[0], splits[1])
        train_dataset = Dataset(src_indexed_dataset, tgt_indexed_dataset,
                                splits[0], splits[1], args.seq_length,
                                args.decoder_seq_length)

        _print_stats('valid', splits[1], splits[2])
        valid_dataset = Dataset(src_indexed_dataset, tgt_indexed_dataset,
                                splits[1], splits[2], args.seq_length,
                                args.decoder_seq_length)
        return train_dataset, valid_dataset

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        if self.sampling is not None:
            idx = self.sampling[self.sampling_indx]
            self.sampling_indx += 1
            if self.sampling_indx == len(self.sampling):
                self.sampling_indx = 0
                self.rng.shuffle(self.sampling)

        local_idx = idx + self.start_index
        assert local_idx < self.end_index
        
        # Truncate input and output sequences to the maximum length
        src_sent = self.src_dataset[local_idx]
        src_tokens = [token for token in src_sent]
        if len(src_tokens) > self.max_encoder_seq_length - 2:
            src_tokens = src_tokens[:self.max_encoder_seq_length - 2]

        tgt_sent = self.tgt_dataset[local_idx]
        tgt_tokens = [token for token in tgt_sent]
        if len(tgt_tokens) > self.max_decoder_seq_length - 2:
            tgt_tokens = tgt_tokens[:self.max_decoder_seq_length - 2]

        enc_ids, tokentypes_enc, enc_seq_length, dec_in_ids, \
        dec_out_ids, dec_seq_length = \
            build_tokens_types_from_ids(
                src_tokens,
                tgt_tokens,
                self.max_seq_length,
                self.decoder_seq_length,
                self.tokenizer.pad,
                self.tokenizer.bos,
                self.tokenizer.eos)
        return (enc_ids, tokentypes_enc, enc_seq_length,
                dec_in_ids, dec_out_ids, dec_seq_length)

    def round_to_nearest(self, length, modulo):
        return (length + modulo - 1) // modulo * modulo

    def _collate_fn(self, batch):
        """Build collate function for data loader."""
        _, _, enc_seq_lengths, _, _, dec_seq_lengths = zip(*batch)
        max_enc_seq_len = self.round_to_nearest(max(enc_seq_lengths), 8)
        max_dec_seq_len = self.round_to_nearest(max(dec_seq_lengths), 8)

        batch_size = len(batch)
        enc_ids = torch.zeros((batch_size, max_enc_seq_len), dtype=torch.int64)
        tokentypes_enc = torch.zeros((batch_size, max_enc_seq_len), dtype=torch.int64)
        dec_in_ids = torch.zeros((batch_size, max_dec_seq_len), dtype=torch.int64)
        labels = torch.zeros((batch_size, max_dec_seq_len), dtype=torch.int64)
        loss_mask = torch.zeros((batch_size, max_dec_seq_len), dtype=torch.int64)

        for i in range(batch_size):
            enc_seq_len = len(batch[i][0])
            enc_ids[i] = torch.tensor(batch[i][0] + [0] * (max_enc_seq_len - enc_seq_len), dtype=torch.int64)
            tokentypes_enc[i] = torch.tensor(batch[i][1] + [0] * (max_enc_seq_len - enc_seq_len), dtype=torch.int64)

            dec_seq_len = len(batch[i][3])
            dec_in_ids[i] = torch.tensor(batch[i][3] + [0] * (max_dec_seq_len - dec_seq_len), dtype=torch.int64)
            labels[i] = torch.tensor(batch[i][4] + [0] * (max_dec_seq_len - dec_seq_len), dtype=torch.int64)
            loss_mask[i] = torch.tensor([1] * dec_seq_len + [0] * (max_dec_seq_len - dec_seq_len), dtype=torch.int64)

        enc_mask = make_attention_mask_3d(enc_ids, enc_ids)
        enc_dec_mask = make_attention_mask_3d(dec_in_ids, enc_ids)
        dec_mask = make_attention_mask_3d(dec_in_ids, dec_in_ids)
        dec_mask = dec_mask * make_history_mask_3d(dec_in_ids)

        sample = {
            'text_enc': enc_ids,
            'text_dec': dec_in_ids,
            'types': tokentypes_enc,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'enc_dec_mask': enc_dec_mask,
        }
        return sample


def build_tokens_types_from_text(src_text, trg_text,
                                 tokenizer, max_seq_length,
                                 decoder_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    src_text_ids = tokenizer.tokenize(src_text)
    trg_text_ids = None
    if trg_text is not None:
        trg_text_ids = tokenizer.tokenize(trg_text)

    return build_tokens_types_from_ids(
        src_text_ids, trg_text_ids,
        max_seq_length, decoder_seq_length,
        tokenizer.pad, tokenizer.bos,
        tokenizer.eos
    )


def build_tokens_types_from_ids(
    src_ids, trg_ids, max_seq_length,
    decoder_seq_length, pad_id,
    bos_id, eos_id
):
    """Build token types and paddings, trim if needed, and pad if needed."""

    enc_ids = []
    tokentypes_enc = []

    # [BOS]
    enc_ids.append(bos_id)
    tokentypes_enc.append(0)

    # A.
    len_src = len(src_ids)
    enc_ids.extend(src_ids)
    tokentypes_enc.extend([0] * len_src)

    # Cap the size.
    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]
        tokentypes_enc = tokentypes_enc[0: max_seq_length - 1]

    # [EOS].
    enc_ids.append(eos_id)
    tokentypes_enc.append(0)

    # B.
    # Not enforcing sequence length checking for target sequences
    # if trg_ids is not None:
    dec_in_ids, dec_out_ids = [bos_id], []
    dec_in_ids.extend(trg_ids)
    dec_out_ids.extend(trg_ids)

    if len(dec_in_ids) > decoder_seq_length:
        dec_in_ids = dec_in_ids[0: decoder_seq_length]
        dec_out_ids = dec_out_ids[0: decoder_seq_length - 1]

    dec_out_ids.append(eos_id)

    return enc_ids, tokentypes_enc, len(enc_ids), dec_in_ids, \
           dec_out_ids, len(dec_in_ids)
