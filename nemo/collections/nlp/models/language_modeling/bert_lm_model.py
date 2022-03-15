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

import os
import json
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss, SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import Perplexity
from nemo.collections.nlp.data.language_modeling.lm_bert_dataset import (
    BertPretrainingDataset,
    BertPretrainingPreprocessedDataloader,
    TarredBertDataset
)
from nemo.collections.nlp.modules.common import BertPretrainingTokenClassifier, SequenceClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ["BERTLMModel"]


class BERTLMModel(ModelPT):
    """
    BERT language model pretraining.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        output_types_dict = {"mlm_log_probs": self.mlm_classifier.output_types["log_probs"]}
        if not self.only_mlm_loss:
            output_types_dict["nsp_logits"] = self.nsp_classifier.output_types["logits"]
        return output_types_dict

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        self.tokenizer = self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=self.register_artifact('language_model.config_file', cfg.language_model.config_file),
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
            vocab_file=self.register_artifact('tokenizer.vocab_file', cfg.tokenizer.vocab_file)
            if cfg.tokenizer
            else None,
        )

        self.hidden_size = self.bert_model.config.hidden_size
        self.vocab_size = self.bert_model.config.vocab_size
        self.only_mlm_loss = cfg.only_mlm_loss

        self.mlm_classifier = BertPretrainingTokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=self.vocab_size,
            num_layers=cfg.num_tok_classification_layers,
            activation="gelu",
            log_softmax=True,
            use_transformer_init=True,
        )

        self.mlm_loss = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id)

        if not self.only_mlm_loss:
            self.nsp_classifier = SequenceClassifier(
                hidden_size=self.hidden_size,
                num_classes=2,
                num_layers=cfg.num_seq_classification_layers,
                log_softmax=False,
                activation="tanh",
                use_transformer_init=True,
            )

            self.nsp_loss = CrossEntropyLoss()
            self.agg_loss = AggregatorLoss(num_inputs=2)

        # # tie weights of MLM softmax layer and embedding layer of the encoder
        if (
            self.mlm_classifier.mlp.last_linear_layer.weight.shape
            != self.bert_model.embeddings.word_embeddings.weight.shape
        ):
            raise ValueError("Final classification layer does not match embedding layer.")
        self.mlm_classifier.mlp.last_linear_layer.weight = self.bert_model.embeddings.word_embeddings.weight
        # create extra bias

        # setup to track metrics
        self.validation_perplexity = Perplexity(compute_on_step=False)

        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
        )
        mlm_log_probs = self.mlm_classifier(hidden_states=hidden_states)
        if self.only_mlm_loss:
            return (mlm_log_probs,)
        nsp_logits = self.nsp_classifier(hidden_states=hidden_states)
        return mlm_log_probs, nsp_logits

    def _compute_losses(self, mlm_log_probs, nsp_logits, output_ids, output_mask, labels):
        mlm_loss = self.mlm_loss(log_probs=mlm_log_probs, labels=output_ids, output_mask=output_mask)
        if self.only_mlm_loss:
            loss, nsp_loss = mlm_loss, None
        else:
            nsp_loss = self.nsp_loss(logits=nsp_logits, labels=labels)
            loss = self.agg_loss(loss_1=mlm_loss, loss_2=nsp_loss)
        return mlm_loss, nsp_loss, loss

    def _parse_forward_outputs(self, forward_outputs):
        if self.only_mlm_loss:
            return forward_outputs[0], None
        else:
            return forward_outputs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        input_ids = input_ids.squeeze(0)
        input_type_ids = input_type_ids.squeeze(0)
        input_mask = input_mask.squeeze(0)
        output_ids = output_ids.squeeze(0)
        output_mask = output_mask.squeeze(0)
        labels = labels.squeeze(0)
        import ipdb; ipdb.set_trace()
        forward_outputs = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        mlm_log_probs, nsp_logits = self._parse_forward_outputs(forward_outputs)
        _, _, loss = self._compute_losses(mlm_log_probs, nsp_logits, output_ids, output_mask, labels)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)
        return {"loss": loss, "lr": lr}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        forward_outputs = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        mlm_log_probs, nsp_logits = self._parse_forward_outputs(forward_outputs)
        _, _, loss = self._compute_losses(mlm_log_probs, nsp_logits, output_ids, output_mask, labels)
        self.validation_perplexity(logits=mlm_log_probs)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """Called at the end of validation to aggregate outputs.

        Args:
            outputs (list): The individual outputs of each validation step.

        Returns:
            dict: Validation loss and tensorboard logs.
        """
        if outputs:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            perplexity = self.validation_perplexity.compute()
            logging.info(f"evaluation perplexity {perplexity.cpu().item()}")
            self.log(f'val_loss', avg_loss)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader(train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader(val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader(test_data_config)


    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            tokenizer_model=cfg.tokenizer_model,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            vocab_file=cfg.vocab_file,
        )
        return tokenizer

    def _setup_dataloader(self, cfg: DictConfig):
        if cfg.get("use_tarred_dataset", False):
            if cfg.get("tar_metadata_file") is None:
                raise FileNotFoundError("Trying to use tarred data set but could not find metadata path in config.")
            metadata_file_list = cfg.get('tar_metadata_file')
            tar_files_list = cfg.get('tar_files', None)
            if isinstance(metadata_file_list, str):
                metadata_file_list = [metadata_file_list]
            if tar_files_list is not None and isinstance(tar_files_list, str):
                tar_files_list = [tar_files_list]
            if tar_files_list is not None and len(tar_files_list) != len(metadata_file_list):
                raise ValueError('The config must have the same number of tarfile paths and metadata file paths.')

            datasets = []
            for idx, metadata_file in enumerate(metadata_file_list):
                with open(metadata_file) as metadata_reader:
                    metadata = json.load(metadata_reader)
                if tar_files_list is None:
                    tar_files = metadata.get('tar_files')
                    if tar_files is not None:
                        # update absolute path of tar files based on metadata_file path
                        valid_tar_files = []
                        metadata_basedir = os.path.abspath(os.path.dirname(metadata_file))
                        updated_fn = 0
                        for fn in tar_files:
                            # if a file does not exist, look in metadata file directory
                            if os.path.exists(fn):
                                valid_fn = fn
                            else:
                                updated_fn += 1
                                valid_fn = os.path.join(metadata_basedir, os.path.basename(fn))
                                if not os.path.exists(valid_fn):
                                    raise RuntimeError(
                                        f"File in tarred dataset is missing from absolute and relative paths {fn}"
                                    )

                            valid_tar_files.append(valid_fn)

                        tar_files = valid_tar_files

                        logging.info(f'Updated the path of {updated_fn} tarred files')
                        logging.info(f'Loading from tarred dataset {tar_files}')
                else:
                    tar_files = tar_files_list[idx]
                    if metadata.get('tar_files') is not None:
                        logging.info(
                            f'Tar file paths found in both cfg and metadata using one in cfg by default - {tar_files}'
                        )
            dataset = TarredBertDataset(
                text_tar_filepaths=tar_files,
                tokenizer = self.tokenizer,
                num_batches=metadata.get('num_batches'),
                shuffle_n=cfg.get("tar_shuffle_n", 100),
                shard_strategy=cfg.get("shard_strategy", "scatter"),
                global_rank=self.global_rank,
                world_size=self.world_size,
            )

            dl = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                sampler=None,
                num_workers=cfg.get("num_workers", 2),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )
        else:
            
            dataset = BertPretrainingDataset(
                tokenizer=self.tokenizer,
                data_file=cfg.data_file,
                max_seq_length=cfg.max_seq_length,
                mask_prob=cfg.mask_prob,
                short_seq_prob=cfg.short_seq_prob,
            )
            dl = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=cfg.batch_size,
                collate_fn=dataset.collate_fn,
                drop_last=cfg.get("drop_last", False),
                shuffle=cfg.shuffle,
                num_workers=cfg.get("num_workers", 0),
            )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="bertbaseuncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/bertbaseuncased/versions/1.0.0rc1/files/bertbaseuncased.nemo",
                description="The model was trained EN Wikipedia and BookCorpus on a sequence length of 512.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="bertlargeuncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/bertlargeuncased/versions/1.0.0rc1/files/bertlargeuncased.nemo",
                description="The model was trained EN Wikipedia and BookCorpus on a sequence length of 512.",
            )
        )
        return result
