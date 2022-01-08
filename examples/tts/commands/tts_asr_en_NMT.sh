python tts_asr.py \
  --tts_model_spectrogram tts_en_fastpitch \
  --tts_model_vocoder tts_squeezewave \
  --asr_model QuartzNet15x5Base-En \
  --input /home/lab/en_de/train.dedup.en \
  --output /home/lab/en_de/train__tts_en_fastpitch__tts_squeezewave__QuartzNet15x5Base-En.en \
  --tmp_dir /media/apeganov/DATA/translation_augmentation_debugging/tmp \
  --num_lines_per_process_for_1_iteration 10000