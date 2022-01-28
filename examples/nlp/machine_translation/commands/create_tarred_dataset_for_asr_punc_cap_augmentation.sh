tokens_in_batch="$1"
num_batches_per_tarfile=50
lines_per_dataset_fragment=200000


python ~/NeMo/examples/nlp/machine_translation/create_tarred_parallel_dataset.py \
    --src_fname "/home/apeganov/en_de_shuffled/augmentation_results/augmented.normalized.en" \
    --tgt_fname "/home/apeganov/en_de_shuffled/train.shuffled.de" \
    --out_dir "/media/apeganov/DATA/ASR_PC_augmentation_of_en_WMT_src/for_upload/train_tarred_${tokens_in_batch}" \
    --encoder_tokenizer_model "/home/apeganov/checkpoints/wmt21_en_de_backtranslated_24x6_averaged_finetuned_13.1.21/071dff1d034e466f935ef6349ede7648_shared_tokenizer.32000.BPE.model" \
    --encoder_tokenizer_bpe_dropout 0 \
    --encoder_tokenizer_coverage 1.0 \
    --decoder_tokenizer_model "/home/apeganov/checkpoints/wmt21_en_de_backtranslated_24x6_averaged_finetuned_13.1.21/071dff1d034e466f935ef6349ede7648_shared_tokenizer.32000.BPE.model" \
    --decoder_tokenizer_bpe_dropout 0 \
    --decoder_tokenizer_coverage 1.0 \
    --tokens_in_batch "${tokens_in_batch}" \
    --num_batches_per_tarfile "${num_batches_per_tarfile}" \
    --lines_per_dataset_fragment "${lines_per_dataset_fragment}" \
    --clean