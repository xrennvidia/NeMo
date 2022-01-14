part="$1"

read -r -d '' command << EOF
tmux new-session -d -s "work" /workspace/NeMo/examples/tts/commands/tts_asr_en_NMT_NGC_citrinet_tmux.sh
EOF

ngc batch run \
  --instance dgx1v.16g.8.norm \
  --name "ml-model.tts_asr ${part}_nmt_data_augmentation" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 94200:/data \
  --commandline "${command}"