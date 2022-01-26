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
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoModel, AutoTokenizer

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.tts.data.datalayers import CTCG2PDataset
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import LabelsType, LossType, MaskType, NeuralType, TokenIndex

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

        # Load appropriate tokenizer from HuggingFace
        self.model_name = cfg.model_name
        print(f"----------> Using model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.max_source_len = cfg.get("max_source_len", self._tokenizer.model_max_length)
        self.max_target_len = cfg.get("max_target_len", self._tokenizer.model_max_length)

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

        # Load pretrained T5 model from HuggingFace
        self.encoder = AutoModel.from_pretrained(self.model_name).encoder
        # add encoder hidden dim size to the config
        self._cfg.decoder.feat_in = self.encoder.config.d_model
        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

    # @typecheck()
    def forward(self, input_ids, attention_mask):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        # hidden_states = [B, seq_len, hid_dim]
        # swap seq_len and hid_dim dimensions
        hidden_states = hidden_states.transpose(1, 2)

        log_probs = self.decoder(encoder_output=hidden_states)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, greedy_predictions

    # ===== Training Functions ===== #
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, input_len, targets, target_lengths = batch
        log_probs, predictions = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        input_len = torch.sum(attention_mask, 1)
        loss = self.loss(log_probs=log_probs, targets=targets, input_lengths=input_len, target_lengths=target_lengths)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        return super().training_epoch_end(outputs)

    # ===== Validation Functions ===== #
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, greedy_predictions = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        input_len = torch.sum(attention_mask, 1)
        val_loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=input_len, target_lengths=target_lengths
        )

        preds_str = self.ctc_decoder_predictions_tensor(greedy_predictions.tolist())
        targets_str = [self.decode_ids_to_str(t) for t in targets.tolist()]

        per = word_error_rate(hypotheses=preds_str, references=targets_str)
        self.log("val_loss", val_loss)
        return {'val_loss': val_loss, 'per': per}

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

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs (reduces across batches, not workers).
        """
        # TODO: Add support for multi GPU
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, sync_dist=True)

        # TODO: Add better PER calculation and logging.
        avg_per = sum([x['per'] for x in outputs]) / len(outputs)
        self.log('val_per', avg_per)
        print(f"---------------> PER: {round(avg_per*100, 2)}%")

        self.log('val_loss', avg_loss, sync_dist=True)
        return {'loss': avg_loss}

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
        self._train_dl = self._setup_dataloader_from_config(cfg, name="train")

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_dataloader_from_config(cfg, name="validation")

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_dataloader_from_config(cfg, name="test")

    # ===== List Available Models - N/A =====$
    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
