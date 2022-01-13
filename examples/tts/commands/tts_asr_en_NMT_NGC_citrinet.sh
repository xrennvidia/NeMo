read -r -d '' command << EOF
set -e -x
cd NeMo
git checkout feat/punc_tarred
git pull
bash reinstall.sh
mkdir -p /raid/tmp
cd examples/tts
python tts_asr.py \
  --tts_model_spectrogram tts_en_fastpitch \
  --tts_model_vocoder tts_squeezewave \
  --asr_model stt_en_citrinet_1024 \
  --input /data/train.en \
  --output /result/train__tts_en_fastpitch__tts_squeezewave__stt_en_citrinet_1024.en \
  --tmp_dir /raid/tmp \
  --num_lines_per_process_for_1_iteration 24000 \
  --cuda_devices 0 1 2 3 4 5 6 7 \
  --asr_batch_size 48 \
  --tts_tokens_in_batch 15000 \
  --resume 2>&1 | tee --append /result/logs.txt
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.8.norm \
  --name "ml-model.tts_asr nmt_data_augmentation" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 93942:/data \
  --commandline "${command}"