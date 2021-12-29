dataset_dir="$(realpath "$1")"
asr_model="$2"  # Path to checkpoint or NGC pretrained name
output_dir="$(realpath "$3")"
kenlm_model="$4"

beam_width_values=(1 2 4 8 16 32 64 128 256)
alpha_values=(0.5 1 1.5 2 2.5 3.5 5 10)
beta_values=(0.5 1 1.5 2 4)

en_ground_truth_manifest="${output_dir}/en_ground_truth_manifest.json"
tmp_transcript_no_numbers="${output_dir}/transcript_no_numbers.manifest"
tmp_transcript="${output_dir}/transcript.manifest"
tmp_wer_file="${output_dir}/tmp_wer.json"
tmp_kenlm_outputs="${output_dir}/tmp_kenlm"

fixed_alpha=$(python -c "print(float(2))")
fixed_beta=$(python -c "print(float(1.5))")

wer_by_beam_width="${output_dir}/wer_by_beam_width_alpha${fixed_alpha}_beta${fixed_beta}.txt"
> "${wer_by_beam_width}"

for bw in "${beam_width_values[@]}"; do
  bw=$(python -c "print(int(${bw}))")
  python ~/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py \
    --nemo_model_file "${asr_model}" \
    --input_manifest "${en_ground_truth_manifest}" \
    --kenlm_model_file "${kenlm_model}" \
    --acoustic_batch_size 1 \
    --beam_width "${bw}" \
    --beam_alpha "${fixed_alpha}" \
    --beam_beta "${fixed_beta}" \
    --preds_output_folder ${tmp_kenlm_outputs} \
    --decoding_mode beamsearch_ngram
  python test_iwslt_and_perform_all_ops_common_scripts/text_to_manifest.py \
    --input "${kenlm_outputs}/preds_out_width${bw}_alpha${fixed_alpha}_beta${fixed_beta}.tsv" \
    --output "${tmp_transcript_no_numbers}" \
    --reference_manifest "${en_ground_truth_manifest}" \
    --take_every_n_line "${bw}"
  python test_iwslt_and_perform_all_ops_common_scripts/text_to_numbers.py \
    -i "${tmp_transcript_no_numbers}" \
    -o "${tmp_transcript}"
  wer="$(python iwslt_scoring/wer_between_2_manifests.py "${tmp_transcript}" "${en_ground_truth_manifest}" -o "${tmp_wer_file}")"
  echo "${bw} ${wer}" >> "${wer_by_beam_width}"
done


fixed_beam_width=$(python -c "print(float(4))")
wer_by_alpha_and_beta="${output_dir}/wer_by_alpha_and_beta_beam_width${fixed_beam_width}.txt"
> "${wer_by_beam_width}"

for alpha in "${alpha_values[@]}"; do
  for beta in "${beta_values[@]}"; do
    alpha=$(python -c "print(int(${alpha}))")
    beta=$(python -c "print(int(${beta}))")
    python ~/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py \
      --nemo_model_file "${asr_model}" \
      --input_manifest "${en_ground_truth_manifest}" \
      --kenlm_model_file "${kenlm_model}" \
      --acoustic_batch_size 1 \
      --beam_width "${fixed_beam_width}" \
      --beam_alpha "${alpha}" \
      --beam_beta "${beta}" \
      --preds_output_folder ${tmp_kenlm_outputs} \
      --decoding_mode beamsearch_ngram
    python test_iwslt_and_perform_all_ops_common_scripts/text_to_manifest.py \
      --input "${tmp_kenlm_outputs}/preds_out_width${fixed_beam_width}_alpha${alpha}_beta${beta}.tsv" \
      --output "${tmp_transcript_no_numbers}" \
      --reference_manifest "${en_ground_truth_manifest}" \
      --take_every_n_line "${fixed_beam_width}"
    python test_iwslt_and_perform_all_ops_common_scripts/text_to_numbers.py \
      -i "${tmp_transcript_no_numbers}" \
      -o "${tmp_transcript}"
    wer="$(python iwslt_scoring/wer_between_2_manifests.py "${tmp_transcript}" "${en_ground_truth_manifest}" -o "${tmp_wer_file}")"
    echo "${alpha} ${beta} ${wer}" >> "${wer_by_alpha_and_beta}"
  done
done