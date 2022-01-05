set -e -x
spectrogram_generators=(tts_en_fastpitch tts_en_tacotron2 tts_en_glowtts)
vocoders=(tts_waveglow_88m tts_squeezewave tts_uniglow tts_melgan tts_hifigan)
e2e_models=(tts_en_e2e_fastpitchhifigan)

output_dir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/test_tts_2_stages
audio_dir="${output_dir}/audio"

wer_file="${output_dir}/wer.txt"
rm -rf "${audio_dir}"/* "${output_dir}"/asr_preds__* "${output_dir}/asr_references.txt" "${wer_file}"

for e2e_model in "${e2e_models}"; do
  python test_tts_infer.py \
    --e2e_model "${e2e_model}" \
    --input ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/iwslt_en_text.txt \
    --asr_preds "${output_dir}/asr_preds__${e2_model}.txt" \
    --asr_references "${output_dir}/asr_references.txt" \
    --audio_preds "${audio_dir}/${e2e_model}" \
    --wer_file "${wer_file}" \
    --no_batching
done

for spectrogram_generator in "${spectrogram_generators[@]}"; do
  for vocoder in "${vocoders[@]}"; do
    python test_tts_infer.py \
      --tts_model_spec "${spectrogram_generator}" \
      --tts_model_vocoder "${vocoder}" \
      --input ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/iwslt_en_text.txt \
      --asr_preds "${output_dir}/asr_preds__${spectrogram_generator}__${vocoder}.txt" \
      --asr_references "${output_dir}/asr_references.txt" \
      --audio_preds "${audio_dir}/${spectrogram_generator}__${vocoder}" \
      --wer_file "${wer_file}" \
      --no_batching
  done
done
set +e +x
