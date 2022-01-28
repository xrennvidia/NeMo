dataset_dir=/home/apeganov/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
result_dir=/home/apeganov/iwslt_2019_test_result/evelina_T5_min_punc_kenlm
mwer_xml="${result_dir}/punc_transcripts_not_segmented_input/stt_en_citrinet_1024_mwer_segmented.xml"
mwer_txt="${result_dir}/punc_transcripts_not_segmented_input/stt_en_citrinet_1024_mwer_segmented.txt"
old_dir="$(pwd)"
cd ~/mwerSegmenter/
(
  __conda_setup="$("/home/${USER}/anaconda3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
    eval "$__conda_setup"
  else
    if [ -f "/home/${USER}/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/${USER}/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/${USER}/anaconda3/bin:$PATH"
    fi
  fi
  unset __conda_setup
  set -e
  conda activate mwerSegmenter  # python 2 conda environment
./segmentBasedOnMWER.sh \
  "${dataset_dir}/src.de" \
  "${dataset_dir}/ref.en" \
  "${result_dir}/punc_transcripts_not_segmented_input/stt_en_citrinet_1024.txt" \
  stt_en_citrinet_1024 \
  English \
  "${mwer_xml}" \
  no \
  1
  conda deactivate
)
cd "${old_dir}"
python ~/NeMo/examples/speech_translation/test_iwslt_and_perform_all_ops_common_scripts/xml_2_text_segs_2_lines.py \
  -i "${mwer_xml}" \
  -o "${mwer_txt}"