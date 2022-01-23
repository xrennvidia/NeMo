work_dir="$1"
if [ -z "$2" ]; then
  model=~/NWInf_results/autoregressive_punctuation_capitalization/intact_punctuation_capitalization_training_on_wiki_wmt_bs80k_steps500k_from_scratch/60k_steps/nemo_experiments/checkpoints/Punctuation_and_Capitalization.nemo
else
  model="$2"
fi
for f in $(find "${work_dir}" -name "*.numbers"); do
  base_name="${f::-8}"
  pickled_features="${base_name}.pickle"
  python punctuate_capitalize_infer.py \
    --input_text "${f}" \
    --output_text "${base_name}.restored" \
    --model_path "${model}" \
    --max_seq_length 128 \
    --step 32 \
    --margin 16 \
    --batch_size 192 \
    --make_queries_contain_intact_sentences \
    --no_all_upper_label \
    --fix_decimals \
    --pickled_features "${pickled_features}"
  rm "$pickled_features"
done
