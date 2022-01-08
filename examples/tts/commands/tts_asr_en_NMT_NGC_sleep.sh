read -r -d '' command << EOF
set -e -x
cd NeMo
git checkout feat/punc_tarred
git pull
bash reinstall.sh
mkdir -p /raid/tmp
sleep 350000
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.8.norm \
  --name "ml-model.tts_asr nmt_data_augmentation" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 93942:/data \
  --commandline "${command}"