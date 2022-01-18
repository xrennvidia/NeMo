tokens_in_batch="$1"

python ~/NeMo/examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py \
  --text /media/apeganov/DATA/punctuation_and_capitalization/simplest/3_128/wiki_wmt_17.01.2022/train/input.txt \
  --labels /media/apeganov/DATA/punctuation_and_capitalization/simplest/3_128/wiki_wmt_17.01.2022/train/bert_labels.txt \
  --output_dir /media/apeganov/DATA/punctuation_and_capitalization/simplest/3_128/wiki_wmt_17.01.2022/for_upload/train_t5_tarred_${tokens_in_batch} \
  --tokens_in_batch="${tokens_in_batch}" \
  --num_batches_per_tarfile 100 \
  --lines_per_dataset_fragment 200000 \
  --tokenizer_name t5-large \
  --use_fast_tokenizer \
  --not_add_cls_and_sep_tokens