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

from typing import Dict, List, Optional

import torch
from torch import nn

from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import ChannelType, NeuralType

__all__ = ['PromptEncoder']


class PromptEncoder(NeuralModule, Exportable):
    """
    The Prompt Encoder network that is used to generate the virtual token embeddings
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
<<<<<<< HEAD
            "template_embeddings": NeuralType(('B', 'T', 'C'), ChannelType(), optional=True),
=======
            "enc_taskname": NeuralType(('B', 'T', 'C'), ChannelType(), optional=True),
>>>>>>> bc6215f166e69502fd7784fc73a5c2c39b465819
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output_embeds": NeuralType(('B', 'T', 'C'), ChannelType())}

    def __init__(self, prompt_seq_len: int, hidden_size: int, lstm_dropout: float, num_layers: int, reparametrize: bool):
        """
        Initializes the PromptEncoder module.
        Args:
            prompt_seq_len: number of prompt embeddings to produce
            hidden_size: hidden dimension
            lstm_dropout: the dropout used for the LSTM
            num_layers: number of layers used in the LSTM
        """
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        self.hidden_size = hidden_size
        self.reparametrize = reparametrize
        # ent embedding
<<<<<<< HEAD
        self.cloze_mask = [1] * prompt_seq_len
=======
        self.cloze_length = template
        self.cloze_mask = [1] * sum(self.cloze_length)
>>>>>>> bc6215f166e69502fd7784fc73a5c2c39b465819
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool()
        self.register_buffer('seq_indices', torch.LongTensor(list(range(len(self.cloze_mask)))))

        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask), self.hidden_size)
<<<<<<< HEAD
        #TODO: add MLP reparametrize here
=======
>>>>>>> bc6215f166e69502fd7784fc73a5c2c39b465819
        # LSTM
        self.lstm_head = torch.nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size)
        )

    @typecheck()
<<<<<<< HEAD
    def forward(self, template_embeddings=None) -> torch.Tensor:
        prompt_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        if self.reparametrize:
            if template_embeddings is not None:
                bz, task_seq, _ = template_embeddings.shape
                _, seq, emb = prompt_embeds.shape
                prompt_embeds = prompt_embeds.expand(bz, seq, emb).clone()
                length = min(task_seq, seq)
                prompt_embeds[:, 0:length, :] = template_embeddings[:, 0:length, :]
            prompt_embeds = self.mlp_head(self.lstm_head(prompt_embeds)[0])
        return prompt_embeds
=======
    def forward(self, enc_taskname) -> torch.Tensor:
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        if enc_taskname is not None:
            bz, task_seq, _ = enc_taskname.shape
            _, seq, emb = input_embeds.shape
            input_embeds = input_embeds.expand(bz, seq, emb).clone()
            length = min(task_seq, seq)
            input_embeds[:, 0:length, :] = enc_taskname[:, 0:length, :]
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        return output_embeds
>>>>>>> bc6215f166e69502fd7784fc73a5c2c39b465819
