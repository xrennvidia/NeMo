python test_seq_len_step_margin.py \
  --model_path ~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_wmt_t5_large_lr5e-6_steps300000/checkpoints/Punctuation_and_Capitalization.nemo \
  --labels ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/labels_iwslt_en_text.txt \
  --source_text ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/text_iwslt_en_text.txt \
  --output_dir ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/evelina_t5_plotting/step \
  --max_seq_length 128 \
  --margin 0  \
  --step 1 2 4 8 12 16 24 32 48 64 96 \
  --cuda_device 0 \
  --no_cls_and_sep_tokens