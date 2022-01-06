python tts_asr.py \
  --tts_model_spectrogram tts_en_fastpitch \
  --tts_model_vocoder tts_squeezewave \
  --asr_model QuartzNet15x5Base-En \
  --input /media/apeganov/DATA/translation_augmentation_debugging/en1000.txt \
  --output /media/apeganov/DATA/translation_augmentation_debugging/output1000.txt \
  --tmp_dir /media/apeganov/DATA/translation_augmentation_debugging/tmp \
  --num_lines_per_process_for_1_iteration 120