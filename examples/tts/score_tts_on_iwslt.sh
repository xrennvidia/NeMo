set -e
spectrogram_generators=(tts_en_tacotron2 tts_en_glowtts tts_en_fastpitch tts_en_talknet)
vocoders=(tts_waveglow_88m tts_squeezewave tts_uniglow tts_melgan tts_hifigan)

output_dir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/test_tts_2_stages

for spectrogram_generator in "${spectrogram_generators[@]}"; do
  for vocoder in "${vocoders[@]}"; do
    python test_tts_infer.py \
      --tts_model_spec "${spectrogram_generator}" \
      --tts_model_vocoder "${vocoder}" \
      --input ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/iwslt_en_text.txt \
      --asr_preds "${output_dir}/asr_preds__${spectrogram_generator}__${vocoder}.txt" \
      --asr_references "${output_dir}/asr_references.txt" \
      --audio_preds "${output_dir}/audio/${spectrogram_generator}__${vocoder}" \
      --wer_file "${output_dir}/wer.txt"
  done
done
set +e
