tokens_in_batch="$1"
num_batches_per_tarfile=50
lines_per_dataset_fragment=200000

work_dir=/media/apeganov/DATA/punctuation_and_capitalization/all_punc_no_u/3_128/wiki_wmt_18.01.2022


python ~/NeMo/examples/nlp/machine_translation/create_tarred_parallel_dataset.py \
    --src_fname "${work_dir}/train/input.txt" \
    --tgt_fname "${work_dir}/train/autoregressive_labels.txt" \
    --out_dir "${work_dir}/for_upload/train_aayn_tarred_${tokens_in_batch}" \
    --encoder_tokenizer_model "${work_dir}/for_upload/input.BPE.25000.model" \
    --encoder_tokenzer_bpe_dropout 0 \
    --encoder_tokenizer_model "${work_dir}/for_upload/autoregressive_labels.BPE.10.model" \
    --decoder_tokenizer_bpe_dropout 0 \
    --decoder_tokenizer_coverage 1.0 \
    --tokens_in_batch "${tokens_in_batch}" \
    --num_batches_per_tarfile "${num_batches_per_tarfile}" \
    --lines_per_dataset_fragment "${lines_per_dataset_fragment}" \
    --clean