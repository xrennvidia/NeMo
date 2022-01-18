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

import argparse
import contextlib
import itertools
import json
import time
from collections import UserList
from typing import Optional

import numpy as np
import torch
import tqdm
from scipy.stats import norm
from torch import nn

from nemo.collections.tts.models import HifiGanModel, MixerTTSModel, UnivNetModel

model_name2model_class = {
    'univ_net': UnivNetModel,
    'mixer_tts': MixerTTSModel,
    'hifi_gan': HifiGanModel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TTS Pipeline Benchmark')
    parser.add_argument('--manifest-path', type=str, required=True)

    parser.add_argument('--spec-gen-name', type=str, required=True)
    parser.add_argument('--spec-gen-pretrained-key', type=str)
    parser.add_argument('--spec-gen-ckpt-path', type=str)

    parser.add_argument('--vocoder-name', type=str, required=True)
    parser.add_argument('--vocoder-pretrained-key', type=str)
    parser.add_argument('--vocoder-ckpt-path', type=str)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cudnn-benchmark', action='store_true', default=False)
    parser.add_argument('--amp-half', action='store_true', default=False)
    parser.add_argument('--amp-autocast', action='store_true', default=False)
    parser.add_argument('--torchscript', action='store_true', default=False)

    parser.add_argument('--warmup-repeats', type=int, default=3)
    parser.add_argument('--n-repeats', type=int, default=10)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-chars', type=int, default=128)
    parser.add_argument('--n-samples', type=int, default=128)

    return parser.parse_args()


def make_data(manifest_path: str, n_chars: Optional[int] = None, n_samples: Optional[int] = None):
    """Makes data source and returns batching functor and total number of samples."""

    if n_chars is None:
        raise ValueError("Unfixed number of input chars is unsupported for now.")

    raw_text_data = []
    with open(manifest_path, 'r') as f:
        for line in f.readlines():
            line_data = json.loads(line)
            raw_text_data.append(line_data['text'])

    if n_samples is None:
        n_samples = len(raw_text_data)

    raw_text_stream = itertools.cycle(raw_text_data)
    data, raw_text_buffer = [], []
    while len(data) < n_samples:
        raw_text_buffer.append(next(raw_text_stream))
        raw_text_from_buffer = ' '.join(raw_text_buffer)
        if len(raw_text_from_buffer) >= n_chars:
            data.append(dict(raw_text=raw_text_from_buffer[:n_chars]))
            raw_text_buffer.clear()

    # This is probably redundant as all samples are of the same length.
    data.sort(key=lambda d: len(d['raw_text']), reverse=True)  # Bigger samples are more important.

    data = {k: [s[k] for s in data] for k in data[0]}
    raw_text_data = data['raw_text']
    total_samples = len(raw_text_data)

    def batching(batch_size):
        """<batch size> => <batch generator>"""
        for i in range(0, len(raw_text_data), batch_size):
            yield dict(raw_text=raw_text_data[i : i + batch_size])

    return batching, total_samples


def load_and_setup_model(
    model_name: str, ckpt_path: str = None, pretrained_key: str = None, amp: bool = False, torchscript: bool = False,
) -> nn.Module:
    model_class = model_name2model_class[model_name]

    if pretrained_key is not None:
        model = model_class.from_pretrained(model_name=pretrained_key)
    elif ckpt_path is not None:
        model = model_class.restore_from(restore_path=ckpt_path)
    else:
        raise ValueError("Either pretrained_key or ckpt_path must be provided.")

    if amp:
        model.half()

    if torchscript:
        model = torch.jit.script(model)

    model.eval()

    return model


class MeasureTime(UserList):
    """Convenient class for time measurement."""

    def __init__(self, *args, cuda=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)


def main():
    """Launches TTS benchmark."""
    args = parse_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark  # noqa

    # Load data
    batching, total_samples = make_data(args.manifest_path, args.n_chars, args.n_samples)

    # Init spectrogram generator
    if args.spec_gen_ckpt_path is None and args.spec_gen_pretrained_key is None:
        raise ValueError("Either --spec-gen-ckpt-path or --spec-gen-pretrained-key must be provided.")

    if args.spec_gen_ckpt_path is not None and args.spec_gen_pretrained_key is not None:
        raise ValueError("Both --spec-gen-ckpt-path and --spec-gen-pretrained-key cannot be provided.")

    spec_generator = load_and_setup_model(
        model_name=args.spec_gen_name,
        ckpt_path=args.spec_gen_ckpt_path,
        pretrained_key=args.spec_gen_pretrained_key,
        amp=args.amp_half,
        torchscript=args.torchscript,
    )
    spec_generator.to(args.device)

    # Init vocoder
    if args.vocoder_ckpt_path is None and args.vocoder_pretrained_key is None:
        raise ValueError("Either --vocoder-ckpt-path or --vocoder-pretrained-key must be provided.")

    if args.vocoder_ckpt_path is not None and args.spec_gen_pretrained_key is not None:
        raise ValueError("Both --vocoder-ckpt-path and --vocoder-pretrained-key cannot be provided.")

    vocoder = load_and_setup_model(
        model_name=args.vocoder_name,
        ckpt_path=args.vocoder_ckpt_path,
        pretrained_key=args.vocoder_pretrained_key,
        amp=args.amp_half,
        torchscript=args.torchscript,
    )
    vocoder.to(args.device)

    sample_rate = spec_generator.cfg.preprocessor.sample_rate
    hop_length = spec_generator.cfg.preprocessor.n_window_stride

    def switch_amp_on():
        """Switches AMP on."""
        return torch.cuda.amp.autocast(enabled=True) if args.amp_autocast else contextlib.nullcontext()

    def batches(batch_size):
        """Batches generator."""
        for b in tqdm.tqdm(
            iterable=batching(batch_size),
            total=(total_samples // batch_size) + int(total_samples % batch_size),
            desc='batches',
        ):
            yield b

    # Warmup
    for _ in tqdm.trange(args.warmup_repeats, desc='warmup'):
        with torch.no_grad(), switch_amp_on():
            for batch in batches(args.batch_size):
                mel = spec_generator.generate_spectrogram(raw_texts=batch['raw_text'])
                _ = vocoder.convert_spectrogram_to_audio(spec=mel)

    # General measurement
    gen_measures = MeasureTime(cuda=(args.device != 'cpu'))
    all_letters, all_frames = 0, 0
    all_utterances, all_samples = 0, 0
    for _ in tqdm.trange(args.n_repeats, desc='repeats'):
        for batch in batches(args.batch_size):
            with torch.no_grad(), switch_amp_on(), gen_measures:
                mel = spec_generator.generate_spectrogram(raw_texts=batch['raw_text'])
                wav = vocoder.convert_spectrogram_to_audio(spec=mel)

            all_letters += sum(len(t) for t in batch['raw_text'])  # <raw text length>
            # TODO(oktai15): Actually, this need to be more precise as samples are of different length?
            all_frames += mel.size(0) * mel.size(2)  # <batch size> * <mel length>

            all_utterances += len(batch['raw_text'])  # <batch size>
            # TODO(oktai15): Same problem as above?
            # <batch size> * <mel length> * <hop length> = <batch size> * <audio length>
            all_samples += mel.size(0) * mel.size(2) * hop_length

    gm = np.sort(np.asarray(gen_measures))
    results = {
        'avg_letters/s': all_letters / gm.sum(),
        'avg_frames/s': all_frames / gm.sum(),
        'avg_latency': gm.mean(),
        'all_samples': all_samples,
        'all_utterances': all_utterances,
        'avg_RTF': all_samples / (all_utterances * gm.mean() * sample_rate),
        '90%_latency': gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std(),
        '95%_latency': gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std(),
        '99%_latency': gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std(),
    }
    for k, v in results.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
