# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig

try:
    from flashfftconv import FlashFFTConv
except ImportError:

    def FlashFFTConv(*args, **kwargs):
        """Not imported: FlashFFTConv. An error will be raised if this is called."""
        raise Exception("Not imported: FlashFFTConv")


try:
    from einops import rearrange
except ImportError:
    raise ImportError("einops is required by the Hyena model but cannot be imported")

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:

    def causal_conv1d_fn(*args, **kwargs):
        """Not imported: causal_conv1d_fn. An error will be raised if this is called."""
        raise ImportError("causal_conv1d is required by the Hyena model but cannot be imported")


from typing import Literal

# CP related utils
import torch.distributed as dist
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default


def _get_zigzag_indices(N, device=None):
    """
    Generates the zigzag indices for rearrangement.
    Args:
        N (int): The total number of chunks.
        device (torch.device): The device on which to create tensors.
    Returns:
        torch.Tensor: The zigzag indices.
    """
    half_N = (N + 1) // 2
    idx1 = torch.arange(half_N, device=device)
    idx2 = torch.arange(N - 1, half_N - 1, -1, device=device)
    zigzag_idx = torch.empty(N, dtype=torch.long, device=device)
    zigzag_idx[0::2] = idx1
    zigzag_idx[1::2] = idx2
    return zigzag_idx


def _get_inverse_zigzag_indices(N, device=None):
    """
    Generates the inverse zigzag indices for rearrangement.
    Args:
        N (int): The total number of chunks.
        device (torch.device): The device on which to create tensors.
    Returns:
        torch.Tensor: The inverse zigzag indices.
    """
    half_N = N // 2
    idx1 = torch.arange(half_N, device=device)
    idx2 = torch.arange(N - 1, half_N - 1, -1, device=device)
    zigzag_idx = torch.empty(N, dtype=torch.long, device=device)
    zigzag_idx[0::2] = idx1
    zigzag_idx[1::2] = idx2
    inverse_zigzag_idx = torch.argsort(zigzag_idx)
    return inverse_zigzag_idx


def all_to_all_single_fn(
    group: dist.ProcessGroup,
    type: Literal["split_to_full", "full_to_split"],
    input: torch.Tensor,
    with_zigzag_splitting: bool = True,
) -> torch.Tensor:
    """
    Autograd-aware all_to_all_single communication function.
    Args:
        group (dist.ProcessGroup): The process group for communication.
        type (str): Either 'split_to_full' or 'full_to_split' to specify the communication pattern.
        input (torch.Tensor): Input tensor to be communicated.
        with_zigzag_splitting (bool, optional): Whether to apply zigzag splitting. Defaults to True.
    Returns:
        torch.Tensor: Output tensor after communication.
    """

    world_size = dist.get_world_size(group=group)

    if type == "split_to_full":
        """Given an split sequence, it gathers the whole sequence, while splitting across the channels dimension."""

        B, D, local_length = input.shape
        L = local_length * world_size
        d = D // world_size

        # Reshape and permute input for communication
        input_reshaped = rearrange(
            input, "B (cp d) l -> cp B d l", cp=world_size
        ).contiguous()  # [cp_world_size, B, d, l]

        # Perform all_to_all_single communication
        output_reshaped = torch.empty_like(input_reshaped)
        dist.all_to_all_single(output_reshaped, input_reshaped, group=group)  # [cp_world_size, B, d, l]

        # Permute and reshape output back to original form
        output = rearrange(output_reshaped, "cp B d l -> B d (cp l)", cp=world_size).contiguous()

        if with_zigzag_splitting:
            num_chunks = 2 * world_size
            unzigzagged_split_length = L // num_chunks  # Length of each small chunk
            device = output.device
            inverse_zigzag_idx = _get_inverse_zigzag_indices(num_chunks, device=device)

            # Vectorized rearrangement using inverse zigzag indices
            output = (
                output.reshape(B, d, num_chunks, unzigzagged_split_length)
                .index_select(dim=-2, index=inverse_zigzag_idx)
                .reshape(B, d, L)
            )

        return output

    elif type == "full_to_split":
        """
        Given a full sequence split across channels, splits across the sequence length while gathering the channels.
        """

        B, d, L = input.shape

        if with_zigzag_splitting:
            num_chunks = 2 * world_size
            chunk_length = L // num_chunks  # Length of each small chunk
            device = input.device
            zigzag_idx = _get_zigzag_indices(num_chunks, device=device)

            # Ensure L is divisible by num_chunks
            if L % num_chunks != 0:
                raise ValueError(f"Sequence length {L} is not divisible by num_chunks {num_chunks}")

            # Vectorized rearrangement using zigzag indices
            input = (
                input.reshape(B, d, num_chunks, chunk_length).index_select(dim=-2, index=zigzag_idx).reshape(B, d, L)
            )

        # Reshape and permute inputs for communication
        input_reshaped = rearrange(
            input, "b d (cp l) -> cp b d l", cp=world_size
        ).contiguous()  # [cp_world_size, b, d, l]

        # Perform all_to_all_single communication
        output_reshaped = torch.empty_like(input_reshaped)
        dist.all_to_all_single(output_reshaped, input_reshaped, group=group)  # [cp_world_size, B, d, l]

        # Permute and reshape outputs back to original form
        output = rearrange(output_reshaped, "cp b d l -> b (cp d) l", cp=world_size).contiguous()

        return output

    else:
        raise ValueError(f"Unknown type {type}")


from torch.autograd.function import Function


class AllToAllSingleFunction(Function):
    """
    A custom autograd function for performing all_to_all_single communication with optional zigzag splitting.
    Attributes:
    - ctx: A context object that stores information for the forward and backward passes.
    - group: The process group for communication.
    - type: The type of communication pattern ('split_to_full' or 'full_to_split').
    - with_zigzag_splitting: A boolean indicating whether to apply zigzag splitting.
    """

    @staticmethod
    def forward(ctx, input_tensor, group, type, with_zigzag_splitting):
        """
        Forward pass for the AllToAllSingleFunction.
        """
        ctx.group = group
        ctx.type = type
        ctx.with_zigzag_splitting = with_zigzag_splitting

        # Detach input_tensor to prevent PyTorch from tracking operations inside the communication
        input_tensor = input_tensor.detach()

        # Perform the communication operation
        output = all_to_all_single_fn(
            group=ctx.group, type=ctx.type, input=input_tensor, with_zigzag_splitting=ctx.with_zigzag_splitting
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the AllToAllSingleFunction.
        """
        # The backward pass will perform the reverse communication
        grad_input = all_to_all_single_fn(
            group=ctx.group,
            type="split_to_full" if ctx.type != "split_to_full" else "full_to_split",
            input=grad_output,
            with_zigzag_splitting=ctx.with_zigzag_splitting,
        )
        # Return the gradient w.r.t. the input_tensor and None for other arguments
        return grad_input, None, None, None


def zigzag_get_overlapping_patches(data, seq_dim, overlap_size):
    """
    Extracts the overlapping patches from data in each rank.
    Arguments:
        data (torch.Tensor): The concatenated data (chunk_a and chunk_b), e.g., [0, 3] , [1, 2] with zigzag and 2 GPUs.
        seq_dim (int): The sequence dimension along which the data is concatenated.
        overlap_size (int): The size of the overlapping patch.
    Returns:
        overlap_a, overlap_b (torch.Tensor): The overlapping chunks from the data. That is the end of the lowest, and
        the beginning of the last, e.g., end for 0 and start for 3.
    """
    assert seq_dim >= 0, "Negative indexes not supported."

    data_shape = list(data.shape)
    modified_shape = list(data.shape)
    modified_shape[seq_dim : seq_dim + 1] = [2, data_shape[seq_dim] // 2]

    reshaped_data = torch.reshape(data, modified_shape)

    # Move the dimension of the chunks to the first position
    # Create a permutation where seq_dim is moved to position 0
    permute_order = list(range(len(reshaped_data.shape)))
    permute_order.insert(0, permute_order.pop(seq_dim))  # Move seq_dim to index 0

    reshaped_data = reshaped_data.permute(dims=permute_order)

    seq_len = reshaped_data.shape[seq_dim + 1]  # Remember that a new dimension was added.
    overlapping_patches = reshaped_data.narrow(
        dim=seq_dim + 1, start=seq_len - overlap_size, length=overlap_size
    )  # Last n elements.
    return overlapping_patches[0], overlapping_patches[1]


class ExchangeOverlappingRegionsCausal(Function):
    """
    A custom autograd function for exchanging overlapping regions between chunks of data in a causal manner.
    The data is split across multiple GPUs using a distributed process group.
    The forward method handles the exchange of overlapping regions between chunks, while the backward
        method computes the gradients.
    Attributes:
    - ctx: A context object that stores information for the forward and backward passes.
    - chunk_a: Chunk to pass to the left.
    - chunk_b: Chunk to pass to the right.
    - group: The CP group
    - group_rank: The rank in the cp_group.
    """

    @staticmethod
    def forward(ctx, chunk_a, chunk_b, group, group_rank):
        """
        Forward pass for the ExchangeOverlappingRegionsCausal function.
        """
        group_ranks = dist.get_process_group_ranks(group)  # Get all global ranks in the cp_group
        group_world_size = len(group_ranks)  # Size of the cp_group

        ctx.group = group
        ctx.group_rank = group_rank
        ctx.group_world_size = group_world_size
        ctx.group_ranks = group_ranks

        # Initialize requests
        reqs = []

        # Exchange overlaps for chunk_a
        if group_rank > 0:
            # Receive overlap from previous rank
            recv_shape = list(chunk_a.shape)
            recv_prev_a = torch.empty(recv_shape, dtype=chunk_a.dtype, device=chunk_a.device)
            req_recv_a = dist.irecv(recv_prev_a, src=group_ranks[group_rank - 1])
            reqs.append(req_recv_a)
        else:
            recv_prev_a = None

        if group_rank < group_world_size - 1:
            # Send overlap to next rank
            req_send_a = dist.isend(chunk_a.contiguous(), dst=group_ranks[group_rank + 1])
            reqs.append(req_send_a)

        # Exchange overlaps for chunk_b
        if group_rank < group_world_size - 1:
            # Receive overlap from next rank
            recv_shape = list(chunk_b.shape)
            recv_next_b = torch.empty(recv_shape, dtype=chunk_b.dtype, device=chunk_b.device)
            req_recv_b = dist.irecv(recv_next_b, src=group_ranks[group_rank + 1])
            reqs.append(req_recv_b)
        else:
            recv_next_b = None

        if group_rank > 0:
            # Send overlap to previous rank
            req_send_b = dist.isend(chunk_b.contiguous(), dst=group_ranks[group_rank - 1])
            reqs.append(req_send_b)

        # Wait for all communication to finish
        for req in reqs:
            req.wait()

        # If no chunks received, use zeros instead (for consistency)
        if recv_prev_a is None:
            recv_prev_a = torch.zeros_like(chunk_a, dtype=chunk_a.dtype, device=chunk_a.device)
        if recv_next_b is None:
            recv_next_b = chunk_a.clone().contiguous()  # Got to receive from the same rank, but previous split.

        return recv_prev_a, recv_next_b

    @staticmethod
    def backward(ctx, grad_chunk_a, grad_chunk_b):
        """
        Backward pass for the ExchangeOverlappingRegionsCausal function.
        """
        # chunk_a, chunk_b = ctx.saved_tensors
        group_rank = ctx.group_rank
        group_world_size = ctx.group_world_size
        group_ranks = ctx.group_ranks

        # Initialize gradients with zeros
        _grad_chunk_a = torch.zeros_like(grad_chunk_a)
        _grad_chunk_b = torch.zeros_like(grad_chunk_b)

        # Initialize requests
        reqs = []

        # Handling grad_chunk_a

        # If rank > 0, send grad_recv_prev_a to rank - 1
        if group_rank > 0:
            req_send_a = dist.isend(grad_chunk_a.contiguous(), dst=group_ranks[group_rank - 1])
            reqs.append(req_send_a)
        else:
            # At rank 0, there's no previous rank to receive from, so we only consider local gradient contributions
            pass  # No action needed

        # If rank < world_size - 1, receive grad_chunk_a from rank + 1
        if group_rank < group_world_size - 1:
            grad_chunk_a_recv = torch.empty_like(grad_chunk_a)
            req_recv_a = dist.irecv(grad_chunk_a_recv, src=group_ranks[group_rank + 1])
            reqs.append(req_recv_a)

        # Handling grad_chunk_b

        # If rank < world_size - 1, send grad_recv_next_b to rank + 1
        if group_rank < group_world_size - 1:
            req_send_b = dist.isend(grad_chunk_b.contiguous(), dst=group_ranks[group_rank + 1])
            reqs.append(req_send_b)

        # If rank > 0, receive grad_chunk_b from rank - 1
        if group_rank > 0:
            grad_chunk_b_recv = torch.empty_like(grad_chunk_b)
            req_recv_b = dist.irecv(grad_chunk_b_recv, src=group_ranks[group_rank - 1])
            reqs.append(req_recv_b)

        # Wait for all communication to finish
        for req in reqs:
            req.wait()

        # Add received gradients
        if group_rank < group_world_size - 1:
            _grad_chunk_a = grad_chunk_a_recv

        if group_rank > 0:
            _grad_chunk_b = grad_chunk_b_recv

        if group_rank == group_world_size - 1:
            _grad_chunk_a = grad_chunk_b  # In the last split, the chunks are exchanged locally.

        return _grad_chunk_a, _grad_chunk_b, None, None, None


# End of CP related functions


def hyena_no_weight_decay_cond(name, param):
    """
    Condition for no weight decay for Hyena parameters.
    """
    # ImplicitModalFilter parameters
    if name.endswith('filter.p') or name.endswith('filter.R') or name.endswith('filter.gamma'):
        no_wd = True

    # ExplicitSingleDecayFilter parameters
    elif name.endswith('filter.h'):
        no_wd = True

    # TODO: Add overrides for other filter types if needed
    #       Alternatively - do something broader, like `if '.filter.' in name` ???

    # ParallelShortHyenaOperator parameters --> The parameters of the internal ParallelCausalDepthwiseConv1d object
    elif name.endswith('short_conv.short_conv_weight'):
        no_wd = True

    # All other parameters - use default MCore behavior:
    # Do not regularize biases and norm parameters
    # (See megatron.core.optimizer._get_pram_groups)
    else:
        no_wd = name.endswith(".bias") or len(param.shape) == 1

    return no_wd


@torch.jit.script
def _mul_sum(y, q):
    """
    Multiply and sum the elements of two tensors along dimension 1.
    """
    return (y * q).sum(dim=1)


def fftconv_func(u, k, D, dropout_mask, gelu=True, k_rev=None, bidirectional=False):
    """Apply a 1D convolution to the input sequence u using the filter k and the shortcut D."""
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    # check if k is less than seqlen
    if k.shape[-1] < seqlen:
        # Pad the filter k to the length of the input sequence u
        k = torch.nn.functional.pad(k, (0, seqlen - k.shape[-1]))

    # bidirectional
    if bidirectional:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

        # split k along the channel dimension
        k, k2 = k.split(k.shape[1] // 2, dim=1)

        # get fft of both k's
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        k2_f = torch.fft.rfft(k2, n=fft_size) / fft_size

        if len(u.shape) > 3:
            k_f = k_f.unsqueeze(1)
            k2_f = k2_f.unsqueeze(1)

        y1 = u_f * k_f
        y2 = u_f.conj() * k2_f.conj()

        y = torch.fft.irfft(y1 + y2, n=fft_size, norm="forward")[..., :seqlen]

    # causal
    else:
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        if k_rev is not None:
            k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
            k_f = k_f + k_rev_f.conj()

        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

        if len(u.shape) > 3:
            k_f = k_f.unsqueeze(1)

        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


class ImplicitModalFilter(nn.Module):
    """
    An implicit modal filter.
    """

    def __init__(
        self,
        d_model,
        order=64,
        L_cache=None,
        gamma_min=0.01,
        gamma_max=0.1,
        lr=None,
    ):
        super().__init__()
        self.order = order
        self.d_model = d_model
        # Do not register into buffer, so it doesn't cast to BF16!
        self.t = rearrange(torch.arange(L_cache, dtype=torch.float32), "L -> 1 1 L").to(
            device=torch.cuda.current_device()
        )  # <- this should be arange
        self.use_cached_t = False
        with get_cuda_rng_tracker().fork():
            gamma = torch.rand(self.d_model, order, dtype=torch.float32) * (gamma_max - gamma_min) + gamma_min
            gamma = gamma.cuda().log()
            self.gamma = nn.Parameter(gamma)

            R = 1e-1 * torch.randn(d_model, order, dtype=torch.float32) / math.sqrt(order)
            self.R = nn.Parameter(R)
            self.p = nn.Parameter(-torch.ones(d_model, order, dtype=torch.float32))
            setattr(self.gamma, 'tensor_model_parallel', True)
            setattr(self.R, 'tensor_model_parallel', True)
            setattr(self.p, 'tensor_model_parallel', True)

    def get_t(self, L):
        """
        Get the t tensor.
        """
        # Assumes L <= L_cache
        if self.use_cached_t:
            return self.t[..., :L]

        t = rearrange(torch.arange(L, dtype=torch.float32, device=self.t.device), "L -> 1 1 L")
        self.t = t
        self.use_cached_t = True

        return t

    def compute_filter(self, L, t):
        """
        Compute the filter for convolution.
        """
        assert (
            t.dtype == torch.float32
        ), f"t must be float32. At lower precision, indexes will be merged together. Current dtype: {t.dtype}"
        # TODO: See if we can get this kernel to stay FP32. We can but it does not work with the distributed optimizer.
        # assert (
        #     self.p.dtype == torch.float32
        # ), f"p must be float32. At lower precision, indexes will be merged together. Current dtype: {self.p.dtype}"
        # assert (
        #     self.gamma.dtype == torch.float32
        # ), ("gamma must be float32. At lower precision, indexes will be merged together. "
        #    f"Current dtype: {self.gamma.dtype}")
        # assert (
        #     self.R.dtype == torch.float32
        # ), f"R must be float32. At lower precision, indexes will be merged together. Current dtype: {self.R.dtype}"

        logp = -torch.exp(self.p.to(torch.float32))
        glogp = logp * torch.exp(self.gamma.to(torch.float32))
        h = torch.exp(glogp[..., None] * t)
        h = torch.einsum('do,dot->dt', self.R.to(torch.float32), h)
        h = h[None]

        return h, None

    def filter(self, L, *args, **kwargs):
        """
        Get t and the convolution filter for t and the requested sequence length.
        """
        t = self.get_t(L)
        h = self.compute_filter(L, t)
        return h

    def forward(self, L, **kwargs):
        """
        Return the final convolutional filter for the requested sequence length.
        """
        return self.filter(L)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias not sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, {'gamma': 0, 'R': 0, 'p': 0}, sharded_offsets)


class ExplicitSingleDecayFilter(nn.Module):
    """
    An explicit single decay filter.
    """

    def __init__(
        self,
        d_model,
        L_cache,
        log_r_min=0,
        log_r_max=2,
        unit_passthrough=False,
        decay_preset="strong",
        small_init=True,
        num_decay_repeats=1,
    ):
        super().__init__()
        with get_cuda_rng_tracker().fork():
            h = torch.randn(d_model, L_cache) / math.sqrt(L_cache)
        assert decay_preset in ["strong", "normal", "weak"]
        if decay_preset == "strong":
            log_r_min = 0
            log_r_max = 2
        elif decay_preset == "normal":
            log_r_min = -1
            log_r_max = 2
        elif decay_preset == "weak":
            log_r_min = -2
            log_r_max = 2

        if small_init:
            h = h * 1e-5
        if unit_passthrough:
            h[:, :1] = 1.0
        self.num_decay_repeats = num_decay_repeats
        self.h = nn.Parameter(h)
        t = torch.linspace(0, 1, L_cache)[None]
        self.log_r_min = log_r_min
        self.log_r_max = log_r_max
        self.model_parallel_rank = get_tensor_model_parallel_rank()
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        global_d_model = d_model * self.model_parallel_size // self.num_decay_repeats
        decay_domain = torch.logspace(log_r_min, log_r_max, global_d_model)[:, None].repeat(self.num_decay_repeats, 1)
        decay_domain = decay_domain[self.model_parallel_rank * d_model : (self.model_parallel_rank + 1) * d_model, :]
        decay = torch.exp((-decay_domain * t).cuda())
        self.register_buffer("decay", decay)
        setattr(self.h, 'tensor_model_parallel', True)
        setattr(self.decay, 'tensor_model_parallel', True)

    def forward(self, L, *args, **kwargs):
        """
        Forward pass for the explicit single decay filter. This returns the filter for the requested sequence length.
        """
        return self.filter(L, *args, **kwargs)

    @torch.compile(mode="max-autotune")
    def filter(self, L, *args, **kwargs):
        """
        Compute the filter as a function of h and decay for the requested sequence length.
        """
        h = self.h[:, :L]
        h = h * self.decay[:, :L]
        return h

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias not sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {
                'h': 0,
                'decay': 0,
            },
            sharded_offsets,
        )


def small_init_init_method(dim):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution.
    """
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim):
    """
    Initialize the weights of the model using the Wang initialization method.
    """
    std = 2 / n_layers / math.sqrt(dim)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def get_init_method(init_method_name, num_layers, hidden_size):
    """
    Gets parameter initialization methods for the linear layers of the model.
    """
    if init_method_name == "wang_init":
        return wang_init_method(num_layers, hidden_size)
    elif init_method_name == "small_init":
        return small_init_init_method(hidden_size)
    else:
        raise NotImplementedError(f"Unknown init method {init_method_name}")


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    with get_cuda_rng_tracker().fork():
        init_method(weight.data)  # modify the data in place


def get_groups_and_group_sizes(hidden_size, num_groups, world_size, expand_factor):
    """
    Get the groups and group sizes for the model.
    """
    width_per_tp_group = divide(hidden_size, world_size)
    num_groups_per_tp = int(divide(num_groups, world_size) * expand_factor)
    group_dim = width_per_tp_group // num_groups_per_tp
    return width_per_tp_group, num_groups_per_tp, group_dim


class ParallelHyenaOperator(nn.Module):
    """
    A class for the ParallelHyenaOperator.
    """

    def __init__(
        self,
        hidden_size,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        init_method,
        operator_type,
        max_sequence_length,
        downsample_factor=1,
        zigzag=True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.transformer_config = transformer_config
        self.hyena_config = hyena_config
        self.operator_type = operator_type
        self.fp16 = transformer_config.fp16
        self.bf16 = transformer_config.bf16
        self.cgcg_dtype = getattr(torch, hyena_config.cgcg_dtype)  # torch.float32

        if self.operator_type == "hyena_medium_conv" and hyena_config.hyena_medium_filter_cls is not None:
            self.hyena_filter_cls = hyena_config.hyena_medium_filter_cls
        else:
            self.hyena_filter_cls = hyena_config.hyena_filter_cls

        self.downsample_factor = downsample_factor
        self.bidirectional = hyena_config.bidirectional
        self.use_hyena_filter = hyena_config.use_hyena_filter
        self.use_slow_heads = hyena_config.use_slow_heads

        self.zigzag = zigzag

        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.model_parallel_rank = get_tensor_model_parallel_rank()

        self.L = max_sequence_length

        if self.operator_type == "hyena_medium_conv":
            self.num_groups = (
                hyena_config.num_groups_hyena_medium
                if hyena_config.num_groups_hyena_medium is not None
                else hyena_config.num_groups_hyena
            )
        elif self.operator_type == "hyena_short_conv":
            self.num_groups = (
                hyena_config.num_groups_hyena_short
                if hyena_config.num_groups_hyena_short is not None
                else hyena_config.num_groups_hyena
            )
        else:
            # default to the global num_groups_hyena
            self.num_groups = hyena_config.num_groups_hyena

        if self.num_groups is None:
            self.num_groups = transformer_config.hidden_size

        world_size: int = get_tensor_model_parallel_world_size()

        self.width_per_tp_group, self.num_groups, self.group_dim = get_groups_and_group_sizes(
            self.hidden_size, self.num_groups, world_size, hyena_config.hyena_width_expansion
        )

        self.short_conv_L = hyena_config.short_conv_L
        self.use_medium_hyena = True if self.operator_type == "hyena_medium_conv" else False
        self.hyena_medium_conv_len = hyena_config.hyena_medium_conv_len

        # TODO: Check which if of these use_* is needed, if any
        self.use_long_conv1d = hyena_config.use_long_conv1d
        self.use_flashfft = hyena_config.use_flashfft
        self.use_cgcg = hyena_config.use_cgcg
        self.is_medium_cgcg = self.use_cgcg and self.use_medium_hyena

        if self.use_flashfft:
            self.fftconv_fn = FlashFFTConv(self.L, dtype=torch.float16 if self.fp16 else torch.bfloat16)

        # TODO: Check which of these filters can be removed
        #       At the moment only "explicit_single_decay" and "implicit_modal" are used
        if self.hyena_filter_cls == "explicit_single_decay":
            self.filter = ExplicitSingleDecayFilter(
                d_model=self.num_groups,
                L_cache=self.hyena_medium_conv_len,
                decay_preset=hyena_config.explicit_filter_decay_preset,
            )
            self.kernel_size = self.hyena_medium_conv_len
        elif self.hyena_filter_cls == "implicit_modal":
            self.filter = ImplicitModalFilter(
                d_model=self.num_groups,
                L_cache=self.L,
                order=hyena_config.hyena_filter_order,
                gamma_min=hyena_config.modal_gamma_min,
                gamma_max=hyena_config.modal_gamma_max,
            )
            self.kernel_size = self.L
        else:
            raise ValueError(f"Unknown hyena filter class: {self.hyena_filter_cls}")

        with get_cuda_rng_tracker().fork():
            self.conv_bias = nn.Parameter(
                torch.empty(
                    self.width_per_tp_group,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
            # Add attribute to prevent automatic casting during model conversion
            setattr(self.conv_bias, 'tensor_model_parallel', True)
            bounds = math.sqrt(1 / self.kernel_size)
            conv_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
            self.conv_bias.data = conv_init_method(self.conv_bias.data)
            self.conv_bias.model_parallel = True
            self.conv_bias.partition_dim = 0
            self.conv_bias.stride = 1

    def forward(self, x1, x2, v, _hyena_use_cp=True):
        """
        Note:
            Input shapes: bs, seq_length, (num_groups, group_size)
            Output shapes: bs, seq_length, num_groups, group_size
        """

        B, L, G, DG = x1.shape

        # CP control
        if _hyena_use_cp:
            cp_group = get_context_parallel_group()
        else:
            cp_group = None

        # downsampled = self.downsample_factor > 1

        # Only permute if not medium cgcg
        if not self.is_medium_cgcg:
            x1 = rearrange(x1, "b l g dg -> b (g dg) l", g=self.num_groups, dg=self.group_dim)
            x2 = rearrange(x2, "b l g dg -> b (g dg) l", g=self.num_groups, dg=self.group_dim)
            v = rearrange(v, "b l g dg -> b (g dg) l", g=self.num_groups, dg=self.group_dim)

        x1, x2, v = x1[..., :L], x2[..., :L], v[..., :L]

        # FIXME: add support post cp refactor
        # if self.downsample_factor > 1:
        #     x1 = x1[..., :: self.downsample_factor]
        #     x2 = x2[..., :: self.downsample_factor]
        #     v = v[..., :: self.downsample_factor]
        #     L = L // self.downsample_factor

        # The kernel length must be adjusted in CP settings
        _L_kernel = L if cp_group is None else L * len(torch.distributed.get_process_group_ranks(cp_group))
        if self.use_medium_hyena:
            h = self.filter(min(self.hyena_medium_conv_len, _L_kernel))
        else:
            h = self.filter(_L_kernel)

        if type(h) == tuple:
            h = h[0]

        conv_bias = self.conv_bias
        local_size = None

        if cp_group is not None and len(torch.distributed.get_process_group_ranks(cp_group)) > 1:

            x1, x2, v = [
                AllToAllSingleFunction.apply(tensor, cp_group, "split_to_full", True) for tensor in [x1, x2, v]
            ]
            # the tensors are now split across channels, but have full length.
            # [ B, H // num_ranks, L]

            rank = torch.distributed.get_rank(cp_group)
            local_size = self.num_groups // get_context_parallel_world_size()

            if isinstance(self.filter, (ImplicitModalFilter)):
                h = h[:, rank * local_size : (rank + 1) * local_size]
            elif isinstance(self.filter, ExplicitSingleDecayFilter):
                h = h[rank * local_size : (rank + 1) * local_size]
            else:
                raise ValueError(f"Kernels of type {self.filter.__class__} have not been verified with CP.")

            local_bias_size = self.width_per_tp_group // get_context_parallel_world_size()
            conv_bias = self.conv_bias[rank * local_bias_size : (rank + 1) * local_bias_size]

        if self.use_long_conv1d:
            h = h.repeat_interleave(self.group_dim, dim=-2)
            z = x2 * v

            z = (
                F.conv1d(z, h[:, None].flip(-1), padding=L - 1, groups=v.shape[1])[..., :L]
                + conv_bias.unsqueeze(-1) * z
            )
            z = z.to(v.dtype)
            z = x1 * z

        else:
            h = h.repeat_interleave(self.group_dim, dim=-2)

            if self.hyena_config.use_flashfft:
                # squeeze h dim (kernel), to get rid of leading 1 dim
                h = h.squeeze(0)
                z = self.fftconv_fn(v, h, x2, x1)
            else:
                z = x2 * v
                # with torch.autocast("cuda"):
                z = fftconv_func(
                    u=z.to(torch.float32),
                    k=h.to(torch.float32),
                    D=conv_bias.to(torch.float32),
                    dropout_mask=None,
                    gelu=False,
                    bidirectional=self.bidirectional,
                )
                z = z.to(v.dtype)
                z = x1 * z

        # if downsampled:
        #     z = z.repeat_interleave(self.downsample_factor, dim=-1)

        # print(
        #   f"[rank={dist.get_rank()}] shape of z = {z.shape} | "
        #   f"num_groups = {self.num_groups}, local_size = {local_size}"
        # )  # DEBUG

        if cp_group is not None and len(torch.distributed.get_process_group_ranks(cp_group)) > 1:
            z = AllToAllSingleFunction.apply(z, cp_group, "full_to_split", True)
            # [ B, H, L // num_ranks]
        return rearrange(z, "b d l -> b l d")

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        Sharded state dictionary for the ParallelHyenaOperator.
        """
        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                'conv_bias': 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
        )
        # Submodules
        for name, module in self.named_children():
            module_sharded_sd = sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)

            sharded_state_dict.update(module_sharded_sd)
        return sharded_state_dict


class ParallelShortHyenaOperator(nn.Module):
    """
    A class for the ParallelShortHyenaOperator.
    """

    def __init__(
        self,
        hidden_size,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        init_method,
        short_conv_class,
        use_fast_causal_conv=False,
        is_mlp=False,  # TODO: Check if needed, only used when using Hyena for the MLP block
        local_init=False,
        use_conv_bias=True,
    ):
        super().__init__()
        self.transformer_config = transformer_config
        self.hyena_config = hyena_config
        self.is_mlp = is_mlp
        self.hidden_size = hidden_size
        self.cgcg_dtype = getattr(torch, hyena_config.cgcg_dtype)
        self.use_cgcg_mlp = hyena_config.use_cgcg_mlp and self.is_mlp
        self.use_cgcg_short = hyena_config.use_cgcg_short and not self.is_mlp
        self.use_custom_hyena_mlp_kernel = hyena_config.use_custom_hyena_mlp_kernel
        self.use_custom_hyena_short_kernel = hyena_config.use_custom_hyena_short_kernel
        self.use_fast_causal_conv = use_fast_causal_conv

        # world_size = mpu.get_model_parallel_world_size() if not local_init else 1
        # world_size: int = torch.distributed.get_world_size() if not local_init else 1

        world_size: int = get_tensor_model_parallel_world_size() if not local_init else 1
        # assert, if using fast_conv_mixer, then the hyena_short_conv_len must be 3
        if use_fast_causal_conv:
            assert hyena_config.hyena_short_conv_len <= 4, "fast_conv_mixer requires hyena_short_conv_len <= 4"

        # for mlp type
        if is_mlp:
            # option to have a different kernel size for the short conv inside the mlp
            if hyena_config.hyena_mlp_len is not None:
                kernel_size = hyena_config.hyena_mlp_len
            else:
                kernel_size = hyena_config.hyena_short_conv_len

            # check for fast causal conv
            if hyena_config.fast_hyena_mlp_conv:
                assert hyena_config.hyena_mlp_len <= 4, "fast_hyena_mlp_conv requires hyena_mlp_len <= 4"
                use_fast_causal_conv = True

            self.pregate = hyena_config.hyena_mlp_pregate
            self.postgate = hyena_config.hyena_mlp_postgate

            self.num_groups = (
                hyena_config.num_groups_hyena_mlp
                if hyena_config.num_groups_hyena_mlp is not None
                else hyena_config.num_groups_hyena
            )

            if self.num_groups is None:
                self.num_groups = transformer_config.hidden_size

            self.num_groups = int(self.num_groups * hyena_config.hyena_mlp_expansion_factor)
        # handle mixer case
        else:

            kernel_size = hyena_config.hyena_short_conv_len
            self.pregate = hyena_config.hyena_short_conv_pregate
            self.postgate = hyena_config.hyena_short_conv_postgate
            self.num_groups = (
                hyena_config.num_groups_hyena_short
                if hyena_config.num_groups_hyena_short is not None
                else hyena_config.num_groups_hyena
            )
            if self.num_groups is None:
                self.num_groups = transformer_config.hidden_size

            self.num_groups = int(self.num_groups * hyena_config.hyena_width_expansion)

        self.width_per_tp_group, self.num_groups, self.group_dim = get_groups_and_group_sizes(
            self.hidden_size, self.num_groups, world_size, hyena_config.hyena_width_expansion
        )

        self.short_conv = short_conv_class(
            self.width_per_tp_group,
            transformer_config,
            hyena_config=hyena_config,
            kernel_size=kernel_size,
            init_method=init_method,
            bias=hyena_config.conv_proj_bias,
            use_fast_causal_conv=use_fast_causal_conv,
            num_groups=self.num_groups,
            repeat_h_dg=False,
            local_init=local_init,
        )

        self.use_conv_bias = use_conv_bias
        if self.use_conv_bias:
            with get_cuda_rng_tracker().fork():
                self.conv_bias = nn.Parameter(
                    torch.empty(
                        self.num_groups,
                        device=torch.cuda.current_device(),
                        dtype=torch.float32,
                    )
                )
                setattr(self.conv_bias, 'tensor_model_parallel', True)
                bounds = math.sqrt(1 / kernel_size)
                conv_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
                self.conv_bias.data = conv_init_method(self.conv_bias.data)
                self.conv_bias.model_parallel = True
                self.conv_bias.partition_dim = 0
                self.conv_bias.stride = 1

    def forward(self, x1, x2, v, _hyena_use_cp=True):
        """
        Note:
            Input shapes: bs, seq_length, (num_groups, group_size)
            Output shapes: bs, seq_length, num_groups, group_size
        """
        B, L, G, DG = x1.shape

        x1 = rearrange(x1, "b l g dg -> b (g dg) l")
        x2 = rearrange(x2, "b l g dg -> b (g dg) l")
        v = rearrange(v, "b l g dg -> b (g dg) l")

        x1, x2, v = x1[..., :L], x2[..., :L], v[..., :L]

        z = x2 * v if self.pregate else v
        if not self.use_conv_bias:
            z = self.short_conv(z, _use_cp=_hyena_use_cp)
        else:
            # maybe handle num_groups
            bias = self.conv_bias.repeat_interleave(self.group_dim, dim=0)
            z = self.short_conv(z, _use_cp=_hyena_use_cp) + rearrange(bias, "h -> 1 h 1") * z  # conv(z) + bias * z

        z = x1 * z if self.postgate else z

        return rearrange(z, "b d l -> b l d")

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        Sharded state dictionary for the ParallelShortHyenaOperator.
        """
        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                'conv_bias': 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
        )
        # Submodules
        for name, module in self.named_children():
            module_sharded_sd = sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)

            sharded_state_dict.update(module_sharded_sd)
        return sharded_state_dict


class ParallelCausalDepthwiseConv1d(nn.Module):
    """
    A class for the ParallelCausalDepthwiseConv1d.
    """

    def __init__(
        self,
        d_model,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        kernel_size,
        init_method,
        bias=False,  # not currently supported
        use_fast_causal_conv=False,
        num_groups=None,  # enables some weight sharing
        repeat_h_dg=True,
        local_init=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.use_bias = bias
        self.use_fast_causal_conv = use_fast_causal_conv
        self.num_groups = num_groups

        if self.num_groups is None:
            self.num_groups = self.d_model

        self.group_dim = self.d_model // self.num_groups

        if self.use_fast_causal_conv:
            assert causal_conv1d_fn is not None, "custom causal conv not installed"
            weight_shape = [self.num_groups, kernel_size]
        # use torch
        else:
            if hyena_config.use_depthwise_short_conv_grouping:
                weight_shape = [self.num_groups, 1, kernel_size]
                self.conv_groups = self.d_model

            else:
                if repeat_h_dg:
                    weight_shape = [self.num_groups, self.group_dim, kernel_size]
                else:
                    weight_shape = [self.num_groups, 1, kernel_size]

                self.conv_groups = self.num_groups

        with get_cuda_rng_tracker().fork():
            self.short_conv_weight = nn.Parameter(
                torch.empty(
                    weight_shape,
                    device=torch.cuda.current_device(),
                    dtype=transformer_config.params_dtype,
                )
            )
            setattr(self.short_conv_weight, 'tensor_model_parallel', True)

            # Use the standard PyTorch Conv1d class init:
            #   https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
            bounds = math.sqrt(1 / hyena_config.short_conv_L)
            conv_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
            if local_init:
                self.short_conv_weight.data = conv_init_method(self.short_conv_weight.data)
            else:
                # Call this on the module because it also modifies module attributes in addition to the data.
                initialize_affine_weight_gpu(self.short_conv_weight, conv_init_method, partition_dim=0)

    def forward(self, x, _use_cp=True):
        """
        Forward pass for the ParallelCausalDepthwiseConv1d.
        """
        assert x.ndim == 3, "Only 3D tensors supported."

        x_shape = x.shape
        weight = self.short_conv_weight
        pad_size = self.kernel_size - 1

        if _use_cp and get_context_parallel_world_size() > 1:

            cp_group = get_context_parallel_group()
            cp_rank = get_context_parallel_rank()

            # Transfer patches across ranks.
            seq_dim = 2  # Last dimension.
            chunk_a, chunk_b = zigzag_get_overlapping_patches(x, seq_dim=seq_dim, overlap_size=pad_size)
            received_a, received_b = ExchangeOverlappingRegionsCausal.apply(chunk_a, chunk_b, cp_group, cp_rank)

            # Pad and rearrange
            x = rearrange(x, "b h (nc s) -> (nc b) h s", nc=2)
            padding = torch.concat([received_a, received_b], dim=0)

            x = torch.concat([padding, x], dim=-1)

        else:
            x = F.pad(x, (pad_size, 0))

        # maybe handle num_groups
        weight = weight.repeat_interleave(self.group_dim, dim=0)

        if self.use_fast_causal_conv:
            y = causal_conv1d_fn(x, weight, bias=None, activation=None)[..., pad_size:]
        else:

            y = F.conv1d(
                x,
                weight,
                bias=None,
                stride=1,
                padding=0,
                groups=self.conv_groups,
            )

        if _use_cp and get_context_parallel_world_size() > 1:
            y = rearrange(y, "(nc b) h s -> b h (nc s)", nc=2)

        assert y.shape == x_shape, f"y.shape = {y.shape} | x.shape = {x_shape}"

        return y

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias not sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {
                'short_conv_weight': 0,
            },
            sharded_offsets,
        )


def make_upper_case(tokens):
    """
    Replace lowercase ASCII characters with uppercase.
    """
    # tokens, labels, loss_mask, attention_mask, position_ids = batch

    lowercase_mask = (tokens >= 97) & (tokens <= 122)
    uppercase_tensor = tokens.clone()
    uppercase_tensor[lowercase_mask] -= 32

    return uppercase_tensor, lowercase_mask


def reweighted_cross_entropy(loss, labels, lowercase_weight=1.0, normalize_per_batch=True):
    """
    Modified for lower case loss reweighting, using the cross_entropy function as a base.

    If normalize_per_batch, loss_weights are normalized by the number of tokens in the batch so
        the magnitude of the loss is not affected by the number of upper/lower case letters
        otherwise, loss_weights are normalized by the number of tokens: combined_loss/len

    performs mean reduction and applies loss_mask
    """

    labels, loss_mask, lowercase_mask = labels[0], labels[1], labels[2]

    upper_loss_mask = loss_mask.bool() & (~lowercase_mask.bool())
    lower_loss_mask = loss_mask.bool() & lowercase_mask.bool()

    loss_weights = torch.zeros_like(loss_mask)
    loss_weights[upper_loss_mask] = 1.0
    loss_weights[lower_loss_mask] = lowercase_weight

    if normalize_per_batch:
        # Get per-microbatch normalization factor
        weight_sum = loss_weights.sum()
        mask_sum = loss_mask.sum()
        weight_normalizer = torch.maximum(weight_sum, torch.ones_like(weight_sum))
        loss_weights = (mask_sum * loss_weights) / weight_normalizer

    # Apply loss weights and loss mask to the loss
    loss = loss * loss_weights * loss_mask

    return loss
