set -e
model=all_punc_no_u_nmt_wiki_wmt_news_crawl_large6x6_bs400000_steps400000_lr2e-4
work_dir="~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model2"
output_dir="${work_dir}/nmt/demonstrate/${model}/margin"
mkdir -p "${output_dir}"
max_seq_length=128
margins=( 0 1 2 4 8 12 16 24 36 48 )
declare -a steps
for margin in "${margins[@]}"; do
  steps+=( $(expr ${max_seq_length} - 2 * ${margin}) )
done

for i in "${!margins[@]}"; do
  margin="${margins["${i}"]}"
  step="${steps["${i}"]}"
  pred_labels="${output_dir}/labels${margin}.txt"
  python punctuate_capitalize_nmt.py \
      --input_text "${work_dir}/text_iwslt_en_text.txt" \
      --output_text "${output_dir}/text${margin}.txt" \
      --output_labels "${pred_labels}" \
      --model_path ~/NWInf_results/autoregressive_punctuation_capitalization/"${model}"/checkpoints/AAYNLarge6x6.nemo \
      --max_seq_length "${max_seq_length}" \
      --step "${step}" \
      --margin "${margin}" \
      --batch_size 80 \
      --no_all_upper_label \
      --add_source_num_words_to_batch \
      --make_queries_contain_intact_sentences

  python compute_metrics.py \
      --hyp ${pred_labels} \
      --ref "${work_dir}/labels_iwslt_en_text.txt" \
      --output "${output_dir}/scores${margin}.json" \
      --normalize_punctuation_in_hyp \
      --reference_evelina_data_format
done
set +e