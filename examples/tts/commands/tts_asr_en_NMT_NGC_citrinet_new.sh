part="$1"

read -r -d '' command << EOF
set -x -e
cd NeMo
git checkout feat/punc_tarred
git pull
bash reinstall.sh
echo "bash reinstall.sh" >> /result/logs.txt
tmux new-session -d -s "work" 'bash'
ls /workspace/NeMo/examples/tts/commands/ >> logs.txt
tmux new-window -n run -t work: 'bash /workspace/NeMo/examples/tts/commands/tts_asr_en_NMT_NGC_citrinet_tmux.sh ${part}'
sleep 1000000
set +x +e
EOF

ngc batch run \
  --instance dgx1v.16g.8.norm \
  --name "ml-model.tts_asr ${part}_nmt_data_augmentation" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 94200:/data \
  --commandline "${command}"