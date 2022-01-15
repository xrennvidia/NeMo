#!/bin/bash
part="$1"
echo "part=\"\$1\"" >> /result/logs.txt
set -x
echo "set -x" >> /result/logs.txt
tmux source /workspace/.tmux.conf
echo "source /workspace/.tmux.conf" >> /result/logs.txt
mkdir -p /raid/tmp
echo "mkdir -p /raid/tmp" >> /result/logs.txt
cd /workspace/NeMo/examples/tts
echo "cd examples/tts" >> /result/logs.txt
output=/result/train__${part}__tts_en_fastpitch__tts_squeezewave__stt_en_citrinet_1024.en
echo "output=/result/train__${part}__tts_en_fastpitch__tts_squeezewave__stt_en_citrinet_1024.en" >> /result/logs.txt
while [ ! -f "${output}" ]; do
  echo "while [ ! -f "${output}" ]; do" >> /result/logs.txt
  python tts_asr.py \
    --tts_model_spectrogram tts_en_fastpitch \
    --tts_model_vocoder tts_squeezewave \
    --asr_model stt_en_citrinet_1024 \
    --input "/data/${part}.en" \
    --output "${output}" \
    --tmp_wav_dir /raid/tmp \
    --tmp_txt_dir /result/tmp \
    --num_lines_per_process_for_1_iteration 24000 \
    --cuda_devices 0 1 2 3 4 5 6 7 \
    --asr_batch_size 48 \
    --tts_tokens_in_batch 15000 \
    --resume 2>&1 | tee --append /result/logs.txt
done
set +x