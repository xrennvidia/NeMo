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

import re
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from apex.transformer import tensor_parallel, parallel_state
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.t0_dataset import (
    T0DatasetBuilder, T0PrimeDatasetBuilder
)
from nemo.collections.nlp.models.language_modeling.megatron.t5_model import t5_position_ids
from nemo.collections.common.metrics.classification_accuracy import ExactStringMatchMetric
from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5FineTuneModel
from nemo.collections.nlp.modules.common.prompt_encoder import PromptEncoder

from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.utils import logging


class MegatronT0Model(MegatronT5FineTuneModel):
    """
    Megatron t0 multitask fine tuning model
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        self.cfg = cfg
        self.acc_metric_dict = torch.nn.ModuleDict()
        self._reduced_loss_buffer = []

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        tokentype_ids=None,
        lm_labels=None,
        enc_hidden_states=None,
        output_enc_hidden_only=False,
    ):
        result = self.model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_decoder_attn_mask=encoder_decoder_attn_mask,
            tokentype_ids=tokentype_ids,
            lm_labels=lm_labels,
            enc_hidden_states=enc_hidden_states,
            output_enc_hidden_only=output_enc_hidden_only,
        )
        if not output_enc_hidden_only:
            return result[0], result[1]
        else:
            return result

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask, task_ids, prompt_ids \
            = self.process_batch(batch)

        output_tensor, encoder_hidden_states = self(
            tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=labels
        )

        loss = self.model.loss_func(loss_mask, output_tensor)
        self.log('train_loss', loss)
        # Reduced loss for logging. Averages the loss across all workers unlike above which is specific to a DDP rank.
        reduced_loss = average_losses_across_data_parallel_group([loss])
        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self._reduced_loss_buffer = []

        return loss

    def get_accuracy(self, predicted_token_ids, labels, task_ids, prompt_ids):
        preds = predicted_token_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for i, (pred, label, guid, prompt_id) in enumerate(zip(preds, labels, task_ids, prompt_ids)):
            if self.model.tokenizer.eos_id in pred:
                idx = pred.index(self.model.tokenizer.eos_id)
                pred = pred[:idx]
            pred = [id for id in pred if id not in self.model.tokenizer.additional_special_tokens_ids]
            label = [id for id in label if id not in self.model.tokenizer.additional_special_tokens_ids]
            pred = self.model.tokenizer.ids_to_text(pred)
            label = self.model.tokenizer.ids_to_text(label)
            key = f'task{guid.item()}_prompt{prompt_id.item()}'
            if not self.acc_metric_dict.__contains__(key):
                self.acc_metric_dict[key] = ExactStringMatchMetric().to(self.device)
            _ = self.acc_metric_dict[key](pred, label)

    def inference_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)

        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask, task_ids, prompt_ids \
            = self.process_batch(batch)

        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask,
            num_tokens_to_generate=10  #TODO: hardcoded 10 is bad here
        )
        self.get_accuracy(predicted_token_ids, labels, task_ids, prompt_ids)

        return {'loss': loss}

    def inference_epoch_end(self, outputs):
        """Uses exact match"""
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        for key in self.acc_metric_dict.keys():
            accuracy = self.acc_metric_dict[key].compute()
            self.acc_metric_dict[key].reset()
            self.log(f'validation_acc_{key}', accuracy)
        self.log('validation_loss', averaged_loss)
        return averaged_loss[0], accuracy

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs_list):
        avg_val_acc = []
        for task_name, outputs in zip(self.validation_task_names, outputs_list):
            val_loss, val_acc = self.inference_epoch_end(outputs)
            self.log('val_loss_%s' % task_name, val_loss, prog_bar=True)
            self.log('val_acc_%s' % task_name, val_acc, prog_bar=True)
            logging.info(f'Validation loss for {task_name}: {val_loss}')
            logging.info(f'Validation accuracy for {task_name}: {val_acc}')
            avg_val_acc.append(val_acc)
        self.log('val_acc', torch.mean(torch.stack(avg_val_acc)), prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.inference_step(batch, batch_idx)

    def test_epoch_end(self, outputs_list):
        avg_test_acc = []
        for task_name, outputs in zip(self.test_task_names ,outputs_list):
            test_loss, test_acc = self.inference_epoch_end(outputs)
            self.log('test_loss_%s' % task_name, test_loss, prog_bar=True)
            self.log('test_acc_%s' % task_name, test_acc, prog_bar=True)
            logging.info(f'Test loss for {task_name}: {test_loss}')
            logging.info(f'Test accuracy for {task_name}: {test_acc}')
            avg_test_acc.append(test_acc)
        self.log('val_acc', torch.mean(torch.stack(avg_test_acc)), prog_bar=True)

    def get_datasetbuilder(self, split, seq_length):
        datasetbuilder = T0DatasetBuilder(
            t0_type=self.cfg.data.t0_type,
            dir_path=self.cfg.data.dir_path,
            max_sampling_size=self.cfg.data.max_sampling_size,
            split=split,
            tokenizer=self.model.tokenizer,
            max_seq_length=seq_length,
            seed=self.cfg.seed,
            buffer_size=self.cfg.data.buffer_size,
            chunk_size=self.cfg.data.chunk_size,
            use_cache=self.cfg.data.use_cache,
            max_samples=getattr(self.cfg.data, "max_samples", None)
        )
        return datasetbuilder

    def build_train_valid_test_datasets(self):
        logging.info('Building train %s datasets.' % self.cfg.data.t0_type)
        self._train_ds = self.get_datasetbuilder('train', self.cfg.data.train_ds.max_seq_length)
        logging.info('Building validation datasets.')
        self._validation_ds = self.get_datasetbuilder('validation', self.cfg.data.validation_ds.max_seq_length)
        logging.info('Building test datasets.')
        self._test_ds = self.get_datasetbuilder('test', self.cfg.data.test_ds.max_seq_length)

        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building T0 datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_data_loader(self, dataset, collate_fn, batch_size, shuffle, num_workers, pin_memory):
        """Buld dataloader given an input dataset."""
        if dataset is None:
            return None
        # Torch dataloader.
        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def setup(self, stage=None):
        """A PTL method to setup the training, validation and test datasets."""
        if stage == 'predict':
            return
        #TODO: megatron_glue_model.py does not have this condition... why?
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            resume_checkpoint_path = self.trainer.checkpoint_connector.resume_checkpoint_path
            if resume_checkpoint_path:
                consumed_samples = int(
                    float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", resume_checkpoint_path)[0])
                )
            else:
                consumed_samples = 0
            self._train_dl = self.build_data_loader(
                self._train_ds.assemble_datasets(),
                collate_fn=self._train_ds.collate_fn,
                batch_size=self.cfg.data.train_ds.batch_size,
                shuffle=False,
                num_workers=self.cfg.data.train_ds.num_workers,
                pin_memory=True,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            dataset_dict = self._validation_ds.assemble_datasets()
            self.validation_task_names = list(dataset_dict.keys())
            self._validation_dl = [self.build_data_loader(
                dataset,
                collate_fn=self._validation_ds.collate_fn,
                batch_size=self.cfg.data.validation_ds.batch_size,
                shuffle=False,
                num_workers=self.cfg.data.validation_ds.num_workers,
                pin_memory=True,
            ) for dataset in dataset_dict.values()]

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            dataset_dict = self._test_ds.assemble_datasets()
            self.test_task_names = list(dataset_dict.keys())
            self._test_dl = [self.build_data_loader(
                dataset,
                collate_fn=self._test_ds.collate_fn,
                batch_size=self.cfg.data.test_ds.batch_size,
                shuffle=False,
                num_workers=self.cfg.data.test_ds.num_workers,
                pin_memory=True,
            ) for dataset in dataset_dict.values()]

    def complete(self, request: Dict):
        #TODO: not sure if I should I keep this
        """
            Autoregressively invokes language model in the inference mode
        Args:	
            request: Dictionary with the following fields
                * prompt: a string which text the model should complete.
                * tokens_to_generate: how many tokens to generate while doing prompt completion.
        Returns:	
            response: A python dictionary with the following fields
                * prompt: original text of the prompt
                * tokenized_prompt: list of (str) tokens from prompt
                * completion: a python dictionary with the following subfields:
                    * tokens: a list of triples (token, token_id, log_prob) comprising completion
                    * text: completion text (as a single string)
                
        """
        response = {}
        self.freeze()
        # naive greedy slow loop
        # TODO: add option for BeamSearchDecoder

        response['prompt'] = request['prompt'][0]
        response['completion'] = {}
        tokens_enc = request['masked_sample']

        response['masked_input'] = ' '.join(self.model.tokenizer.ids_to_tokens(tokens_enc[0]))
        enc_mask = self.make_inference_attention_mask_3d(tokens_enc, tokens_enc, self.tokenizer.pad_id)
        enc_mask = enc_mask < 0.5

        predicted_tokens_ids, log_probs = self.decode(tokens_enc, enc_mask, int(request['tokens_to_generate']))
        predicted_tokens_ids = predicted_tokens_ids.cpu().numpy()[0].tolist()
        log_probs = log_probs.cpu().numpy()[0].tolist()
        if self.model.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.model.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        else:
            predicted_tokens_ids = [id for id in predicted_tokens_ids if id != self.model.tokenizer.pad_id]
        predicted_tokens_dec = self.mdoel.tokenizer.ids_to_tokens(predicted_tokens_ids)
        response['completion']['text'] = self.model.tokenizer.tokens_to_text(predicted_tokens_dec)
        response['completion']['tokens'] = list(zip(predicted_tokens_ids, predicted_tokens_dec, log_probs))
        self.unfreeze()
        return response

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    def process_batch(self, batch):
        """Build the batch."""

        keys = [
            'text_enc', 'text_dec', 'template', 'labels',
            'loss_mask', 'enc_mask', 'dec_mask', 'enc_dec_mask',
            'task_ids', 'prompt_ids'
        ]
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask'] < 0.5
        dec_mask = data_b['dec_mask'] < 0.5
        enc_dec_mask = data_b['enc_dec_mask'] < 0.5

        # T0 specific
        task_ids = data_b['task_ids'].long()
        prompt_ids = data_b['prompt_ids'].long()

        return tokens_enc, tokens_dec, loss_mask, labels, \
               enc_mask, dec_mask, enc_dec_mask, task_ids, prompt_ids


class MegatronT0PrimeModel(MegatronT0Model):
    """
        Megatron t0 prime multitask fine tuning model using differentiable promtps
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)

        self.hidden_size = self.model.cfg.hidden_size
        self.prompt_seq_len = cfg.prompt_encoder.seq_len
        self.float_type = self.model.model.language_model.encoder.layers[0].dtype

        if cfg.freeze:
            self.model.freeze()

        self.prompt_encoder = PromptEncoder(
            prompt_seq_len=self.prompt_seq_len,
            hidden_size=self.hidden_size,
            prompt_dropout=cfg.prompt_encoder.dropout,
            num_layers=cfg.prompt_encoder.num_layers,
            reparametrize=cfg.prompt_encoder.reparametrize,
            prompt_gen_type=cfg.prompt_encoder.prompt_gen_type,
            trainer=trainer
        )
        self.embeddings = self.model.model.language_model.embedding.word_embeddings
        self.model.tokenizer.add_special_tokens({'additional_special_tokens': [cfg.pseudo_token]})
        self.pseudo_token_id = self.model.tokenizer.token_to_id(cfg.pseudo_token)
        self.pad_token_id = self.model.tokenizer.pad_id if self.model.tokenizer.pad_id is not None \
            else self.model.tokenizer.unk_id

    def embed_input(self, encoder_input_ids, prompt_input_ids):
        bz,seq_len = encoder_input_ids.shape
        queries_for_embedding = encoder_input_ids.clone()

        queries_for_embedding[(encoder_input_ids == self.pseudo_token_id)] = self.pad_token_id
        query_embeddings = self.embeddings(queries_for_embedding).clone().type(self.float_type)
        if self.cfg.prompt_encoder.task_dependent and self.cfg.data.split_template:
            prompt_condition = self.embeddings(prompt_input_ids)
        elif self.cfg.prompt_encoder.task_dependent:
            prompt_condition = query_embeddings
        else:
            prompt_condition = None

        index = (
            (encoder_input_ids == self.pseudo_token_id).nonzero().reshape((bz, -1, 2))[:, :, 1][:, :, None]
        )  # bz
        _, seq, _ = index.shape
        _, _, emb = query_embeddings.shape
        index = index.expand(bz, seq, emb)[:, :self.prompt_seq_len, :]

        if self.float_type == torch.float32:
            replace_embeds = self.prompt_encoder(prompt_condition=prompt_condition)
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                replace_embeds = self.prompt_encoder(prompt_condition=prompt_condition)

        if prompt_condition is None:
            _, replace_seq, _ = replace_embeds.shape
            replace_embeds = replace_embeds.expand(bz, replace_seq, emb)

        query_embeddings.scatter_(1, index, replace_embeds)

        encoder_position_ids = t5_position_ids(encoder_input_ids)
        position_embeddings = self.model.model.language_model.embedding.position_embeddings(encoder_position_ids)
        return query_embeddings + position_embeddings

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        prompt_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        tokentype_ids=None,
        lm_labels=None,
        enc_hidden_states=None,
        output_enc_hidden_only=False,
    ):
        encoder_input = self.embed_input(encoder_input_ids, prompt_input_ids)

        if self.float_type == torch.float32:
            result = self.model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attn_mask=encoder_attn_mask,
                decoder_attn_mask=decoder_attn_mask,
                encoder_decoder_attn_mask=encoder_decoder_attn_mask,
                tokentype_ids=tokentype_ids,
                lm_labels=lm_labels,
                enc_hidden_states=enc_hidden_states,
                output_enc_hidden_only=output_enc_hidden_only,
                encoder_input=encoder_input
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                result = self.model(
                    encoder_input_ids=encoder_input_ids,
                    decoder_input_ids=decoder_input_ids,
                    encoder_attn_mask=encoder_attn_mask,
                    decoder_attn_mask=decoder_attn_mask,
                    encoder_decoder_attn_mask=encoder_decoder_attn_mask,
                    tokentype_ids=tokentype_ids,
                    lm_labels=lm_labels,
                    enc_hidden_states=enc_hidden_states,
                    output_enc_hidden_only=output_enc_hidden_only,
                    encoder_input=encoder_input
                )
        if not output_enc_hidden_only:
            return result[0], result[1]
        else:
            return result

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask, task_ids, prompt_ids, tokens_prompt \
            = self.process_batch(batch)

        output_tensor, encoder_hidden_states = self(
            tokens_enc, tokens_dec, tokens_prompt,
            enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=labels
        )

        loss = self.model.loss_func(loss_mask, output_tensor)
        self.log('train_loss', loss)
        # Reduced loss for logging. This averages the loss across all workers unlike "loss" above which is specific to a DDP rank.
        reduced_loss = average_losses_across_data_parallel_group([loss])
        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self._reduced_loss_buffer = []

        return loss

    def inference_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)

        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask, task_ids, prompt_ids, tokens_prompt \
            = self.process_batch(batch)
        encoder_input = self.embed_input(encoder_input_ids=tokens_enc, prompt_input_ids=tokens_prompt)
        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask, encoder_input=encoder_input,
            num_tokens_to_generate=10  #TODO: hardcoded 10 is bad here
        )
        self.get_accuracy(predicted_token_ids, labels, task_ids, prompt_ids)

        return {'loss': loss}

    def get_datasetbuilder(self, split, seq_length):
        datasetbuilder = T0PrimeDatasetBuilder(
            t0_type=self.cfg.data.t0_type,
            dir_path=self.cfg.data.dir_path,
            max_sampling_size=self.cfg.data.max_sampling_size,
            split=split,
            tokenizer=self.model.tokenizer,
            max_seq_length=seq_length,
            seed=self.cfg.seed,
            buffer_size=self.cfg.data.buffer_size,
            chunk_size=self.cfg.data.chunk_size,
            use_cache=self.cfg.data.use_cache,
            max_samples=getattr(self.cfg.data, "max_samples", None),
            prompt_token_id=self.pseudo_token_id,
            prompt_seq_len=self.prompt_seq_len,
            split_template=self.cfg.data.split_template
        )
        return datasetbuilder

    def process_batch(self, batch):
        """Build the batch."""

        keys = [
            'text_enc', 'text_dec', 'template', 'labels',
            'loss_mask', 'enc_mask', 'dec_mask', 'enc_dec_mask',
            'task_ids', 'prompt_ids'
        ]
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask'] < 0.5
        dec_mask = data_b['dec_mask'] < 0.5
        enc_dec_mask = data_b['enc_dec_mask'] < 0.5

        # T0 specific
        task_ids = data_b['task_ids'].long()
        prompt_ids = data_b['prompt_ids'].long()

        # T0' specific
        tokens_prompt = data_b['template'].long()

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, \
               enc_dec_mask, task_ids, prompt_ids, tokens_prompt