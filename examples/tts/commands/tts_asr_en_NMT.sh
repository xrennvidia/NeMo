python tts_asr.py \
  --tts_model_spectrogram tts_en_fastpitch \
  --tts_model_vocoder tts_squeezewave \
  --asr_model QuartzNet15x5Base-En \
  --input /home/apeganov/en_de_shuffled_4M/train.en \
  --output /home/apeganov/en_de_shuffled_4M/train__tts_en_fastpitch__tts_squeezewave__QuartzNet15x5Base-En.en \
  --tmp_dir /home/apeganov/en_de_shuffled_4M/tmp \
  --num_lines_per_process_for_1_iteration 50000