set -e
model=all_punc_no_u_nmt_wiki_wmt_news_crawl_large6x6_bs400000_steps400000_lr2e-4
work_dir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model2
output_dir="${work_dir}/nmt/demonstrate/${model}/in_practice"
mkdir -p "${output_dir}"
margin=16
step=8
pred_labels="${output_dir}/labels_margin${margin}_step${step}.txt"
python punctuate_capitalize_nmt.py \
    --input_text "${work_dir}/text_iwslt_en_text.txt" \
    --output_text "${output_dir}/text_margin${margin}_step${step}.txt" \
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
    --ref "${work_dir}/all_punc_no_u/autoregressive_labels.txt" \
    --output "${output_dir}/scores_margin${margin}_step${step}.json"
set +e