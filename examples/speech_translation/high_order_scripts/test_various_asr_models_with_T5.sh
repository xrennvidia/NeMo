asr_models=(
  stt_en_citrinet_1024
  ~/checkpoints/CitriNet-1024-8x-Stride-Gamma-0.25.nemo
  ~/checkpoints/Citrinet-1024-SPE-Unigram-1024-Jarvis-ASRSet-2.0_no_weight_decay_e250-averaged.nemo
  QuartzNet15x5Base-En
  stt_en_citrinet_256
  stt_en_citrinet_512
  stt_en_jasper10x5dr
)

for asr_model in "${asr_models[@]}"; do
  asr_model_name="$(basename "${asr_model}")"
  bash test_iwslt.sh \
    ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019 \
    "${asr_model}" \
    ~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_wmt_t5_large_lr5e-6_steps300000/checkpoints/Punctuation_and_Capitalization.nemo \
    ~/checkpoints/wmt21_en_de_backtranslated_24x6_averaged.nemo \
    ~/iwslt_2019_test_result/varying_asr_for_evelina_T5_min_punc_kenlm/${asr_model_name} \
    0 \
    1 \
    0 \
    1 \
    0 \
    1 \
    /media/apeganov/DATA/punctuation_and_capitalization/simplest/92_128/wmt_92_128_14.12.2021/train/ngram6.arpa
  bash test_iwslt.sh \
    ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019 \
    "${asr_model}" \
    ~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_wmt_t5_large_lr5e-6_steps300000/checkpoints/Punctuation_and_Capitalization.nemo \
    ~/checkpoints/wmt21_en_de_backtranslated_24x6_averaged.nemo \
    ~/iwslt_2019_test_result/varying_asr_for_evelina_T5_min_punc/${asr_model_name} \
    0 \
    1 \
    0 \
    1 \
    0 \
    1
done