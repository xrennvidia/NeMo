WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
mkdir -p /result/nemo_experiments
cd NeMo
git checkout feat/punc_tarred
cd examples/nlp/token_classification
wandb login ${WANDB_API_KEY}
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased'); tokenizer.save_pretrained('pretrained_tokenizer')"
python punctuation_capitalization_train.py --config-path=conf \
    --config-name wmt_train_from_scratch \
    trainer.gpus=1
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.8.norm \
  --name "ml-model.bert intact_punctuation_capitalization_training_on_wiki_wmt" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 88512:/data \
  --commandline "${command}"