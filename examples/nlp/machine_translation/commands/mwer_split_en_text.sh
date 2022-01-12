dataset_dir=/home/apeganov/iwslt_2019_test_result
mwer_xml=/home/apeganov/iwslt_2019_test_result/AAYNLarge6x6_all_punc_no_u_kenlm/punc_transcripts_not_segmented_input/stt_en_citrinet_1024_mwer_segmented.xml
mwer_txt=/home/apeganov/iwslt_2019_test_result/AAYNLarge6x6_all_punc_no_u_kenlm/punc_transcripts_not_segmented_input/stt_en_citrinet_1024_mwer_segmented.txt
old_dir="$(pwd)"
cd ~/mwerSegmenter/
./segmentBasedOnMWER.sh \
  "${dataset_dir}/IWSLT.TED.tst2019.en-de.de.xml" \
  "${dataset_dir}/IWSLT.TED.tst2019.en-de.en.xml" \
  /home/apeganov/iwslt_2019_test_result/AAYNLarge6x6_all_punc_no_u_kenlm/punc_transcripts_not_segmented_input/stt_en_citrinet_1024.txt \
  stt_en_citrinet_1024 \
  Enlish \
  /home/apeganov/iwslt_2019_test_result/AAYNLarge6x6_all_punc_no_u_kenlm/punc_transcripts_not_segmented_input/stt_en_citrinet_1024_mwer_segmented.xml \
  no \
  1
cd old_dir
python ~/NeMo/examples/speech_translation/test_iwslt_and_perform_all_ops_common_scripts/xml_2_text_segs_2_lines.py \
  -i "${mwer_xml}" \
  -o "${mwer_txt}"