#!/bin/bash
start="$1"
end="$2"
set -x
tmux source /workspace/.tmux.conf
mkdir -p /raid/tmp
cd /workspace/NeMo/examples/tts
for i in $(seq "${start}" 1 "${end}"); do
  output="/result/${i}.txt"
  normed_output="/raid/normed${i}.txt"
  tmp_normed=/raid/tmp_norm
  python normalize_parallel.py \
    --input_file "/data/${i}.txt" \
    --tmp_dir "${tmp_normed}" \
    --output_file "${normed_output}"
  while [ ! -f "${output}" ]; do
    echo "while [ ! -f "${output}" ]; do" >> /result/logs.txt
    python tts_asr.py \
      --tts_model_spectrogram tts_en_fastpitch \
      --tts_model_vocoder tts_squeezewave \
      --asr_model stt_en_citrinet_1024 \
      --input "${normed_output}" \
      --output "${output}" \
      --tmp_wav_dir /raid/tmp \
      --tmp_txt_dir /result/tmp \
      --num_lines_per_process_for_1_iteration 24000 \
      --cuda_devices 0 1 2 3 4 5 6 7 \
      --asr_batch_size 48 \
      --tts_tokens_in_batch 15000 \
      --resume 2>&1 | tee --append /result/logs.txt
  done
done
set +x