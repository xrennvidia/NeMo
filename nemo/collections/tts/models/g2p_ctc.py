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

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer
from nemo.collections.tts.data.datalayers import CTCG2PDataset
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import LabelsType, LossType, MaskType, NeuralType, TokenIndex
from nemo.utils import logging

__all__ = ['CTCG2PModel']


@dataclass
class CTCG2PConfig:
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class CTCG2PModel(ModelPT):  # TODO: Check parent class
    """
    CTC-based grapheme-to-phoneme model.
    """

    # @property
    # def input_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return {
    #         "input_ids": NeuralType(('B', 'T'), TokenIndex()),
    #         "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
    #         "labels": NeuralType(('B', 'T'), LabelsType()),
    #     }
    #
    # @property
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return {"loss": NeuralType((), LossType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        if "t5" in cfg.model_name.lower():
            self.mode = "t5"
        elif "conformer" in cfg.model_name.lower():
            self.mode = "conformer"
        else:
            raise ValueError(f"'T5' or 'Conformer' name must be added to 'model.model_name'")

        if self.mode == "t5":
            ### T5
            # Load appropriate tokenizer from HuggingFace
            print(f"----------> Using model: {cfg.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

            self.max_source_len = cfg.get("max_source_len", self._tokenizer.model_max_length)
            self.max_target_len = cfg.get("max_target_len", self._tokenizer.model_max_length)
        else:
            chars = [
                " ",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "'",
            ]
            vocab_file = "/tmp/char_vocab.txt"
            with open(vocab_file, "w") as f:
                [f.write(f'"{ch}"\n') for ch in chars]

            self._tokenizer = CharTokenizer(vocab_file=vocab_file)
            self.max_source_len = cfg.get("max_source_len", 512)
            self.max_target_len = cfg.get("max_target_len", 512)

        # Ensure passed cfg is compliant with schema
        schema = OmegaConf.structured(CTCG2PConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        OmegaConf.merge(cfg, schema)

        self.vocabulary = cfg.decoder.vocabulary
        self.labels_tkn2id = {l: i for i, l in enumerate(cfg.decoder.vocabulary)}
        self.labels_id2tkn = {i: l for i, l in enumerate(cfg.decoder.vocabulary)}
        super().__init__(cfg, trainer)

        if self.mode == "t5":
            config = AutoConfig.from_pretrained(cfg.model_name)
            if self._cfg.dropout is not None:
                config.dropout_rate = self._cfg.dropout
                print(f"\nDROPOUT: {config.dropout_rate}")
            self.encoder = AutoModel.from_pretrained(cfg.model_name, config=config).encoder
            # add encoder hidden dim size to the config
            if self.cfg.decoder.feat_in is None:
                self._cfg.decoder.feat_in = self.encoder.config.d_model
        else:
            self.embedding = nn.Embedding(
                embedding_dim=cfg.embedding.d_model, num_embeddings=self._tokenizer.vocab_size, padding_idx=0
            )
            # setup encoder after init()
            self.encoder = EncDecCTCModel.from_config_dict(self._cfg.encoder)
            with open_dict(self._cfg):
                if "feat_in" not in self._cfg.decoder or (
                    not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
                ):
                    self._cfg.decoder.feat_in = self.encoder._feat_out
                if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                    raise ValueError("param feat_in of the decoder's config is not set!")

        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

    # @typecheck()
    def forward(self, input_ids, attention_mask, input_len):
        if self.mode == "t5":
            encoded_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoded_len = torch.sum(attention_mask, 1)
            # encoded_input = [B, seq_len, hid_dim]
            # swap seq_len and hid_dim dimensions to get [B, hid_dim, seq_len]
            encoded_input = encoded_input.transpose(1, 2)
        else:
            input_embedding = self.embedding(input_ids)
            input_embedding = input_embedding.transpose(1, 2)
            encoded_input, encoded_len = self.encoder(audio_signal=input_embedding, length=input_len)

        log_probs = self.decoder(encoder_output=encoded_input)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        import pdb;
        pdb.set_trace()
        return log_probs, greedy_predictions, encoded_len

    # ===== Training Functions ===== #
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )


        loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        return super().training_epoch_end(outputs)

    # ===== Validation Functions ===== #
    def validation_step(self, batch, batch_idx, split="val"):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, greedy_predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )
        val_loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )

        preds_str = self.ctc_decoder_predictions_tensor(greedy_predictions.tolist())
        targets_str = [self.decode_ids_to_str(t) for t in targets.tolist()]
        per = word_error_rate(hypotheses=preds_str, references=targets_str)

        self.log(f"{split}_loss", val_loss)
        return {f"{split}_loss": val_loss, "per": per}

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, split="test")

    def decode_ids_to_str(self, ids: List[int]) -> str:
        blank_id = len(self.labels_id2tkn)
        return " ".join([self.labels_id2tkn[t] for t in ids if t != blank_id])

    def ctc_decoder_predictions_tensor(self, predictions: List[List[int]], predictions_len=None) -> List[str]:
        """
        Decodes a sequence of labels to words

        Args:
            predictions: An integer torch.Tensor of shape [Batch, Time] (if ``batch_index_dim == 0``) or [Time, Batch]
                (if ``batch_index_dim == 1``) of integer indices that correspond to the index of some character in the
                label set.
            predictions_len: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.
            return_hypotheses: Bool flag whether to return just the decoding predictions of the model
                or a Hypothesis object that holds information such as the decoded `text`,
                the `alignment` of emited by the CTC Model, and the `length` of the sequence (if available).
                May also contain the log-probabilities of the decoder (if this method is called via
                transcribe())

        Returns:
            Either a list of str which represent the CTC decoded strings per sample,
            or a list of Hypothesis objects containing additional information.
        """
        blank_id = len(self.labels_id2tkn)
        hypotheses = []

        # iterate over batch
        for ind, prediction in enumerate(predictions):
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]

            # CTC decoding procedure
            decoded_prediction = []
            previous = blank_id
            for p in prediction:
                if (p != previous or previous == blank_id) and p != blank_id:
                    decoded_prediction.append(p)
                previous = p

            text = self.decode_ids_to_str(decoded_prediction)
            hypotheses.append(text)
        return hypotheses

    def validation_epoch_end(self, outputs, split="val"):
        """
        Called at the end of validation to aggregate outputs (reduces across batches, not workers).
        """
        # TODO: Add support for multi GPU
        avg_loss = torch.stack([x[f"{split}_loss"] for x in outputs]).mean()
        self.log(f"{split}_loss", avg_loss, prog_bar=True)

        # TODO: Add better PER calculation and logging.
        avg_per = sum([x["per"] for x in outputs]) / len(outputs)
        self.log(f"{split}_per", avg_per)
        logging.info(f"---------------> PER: {round(avg_per*100, 2)}%")

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, split="test")

    @torch.no_grad()
    def _generate_predictions(self, input_ids: torch.Tensor, model_max_target_len: int = 512):
        """
        Generates predictions and converts IDs to text.
        """
        outputs = self.model.generate(
            input_ids, output_scores=True, return_dict_in_generate=True, max_length=model_max_target_len
        )

        generated_ids, sequence_toks_scores = outputs['sequences'], outputs['scores']
        generated_texts = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_texts, generated_ids, sequence_toks_scores

    # ===== Dataset Setup Functions ===== #
    def _setup_dataloader_from_config(self, cfg, name):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")

        dataset = CTCG2PDataset(
            cfg.dataset.manifest_filepath, self._tokenizer, self.vocabulary, self.max_source_len, self.max_target_len
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        if not cfg or not os.path.exists(cfg.dataset.manifest_filepath):
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg, name="train")

    def setup_validation_data(self, cfg):
        if not cfg or not os.path.exists(cfg.dataset.manifest_filepath):
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg, name="validation")

    def setup_test_data(self, cfg):
        if not cfg or not os.path.exists(cfg.dataset.manifest_filepath):
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg, name="test")

    # ===== List Available Models - N/A =====$
    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
