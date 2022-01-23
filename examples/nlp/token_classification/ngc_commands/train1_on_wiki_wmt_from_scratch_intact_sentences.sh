WANDB_API_KEY="$1"

exp_name="intact_punctuation_capitalization_training_on_wiki_wmt_bs80k_steps500k"
gpus=8

read -r -d '' command << EOF
set -e -x
result_dir=/result/nemo_experiments
mkdir -p "${result_dir}"
cd NeMo
git checkout feat/punc_tarred
git pull
bash reinstall.sh
cd examples/nlp/token_classification
wandb login ${WANDB_API_KEY}
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased'); tokenizer.save_pretrained('pretrained_tokenizer')"
python punctuation_capitalization_train_evaluate.py --config-path=conf/wiki_wmt \
    --config-name local_base_intact_sentences_bs20k_lr1e-4_steps500k \
    exp_manager.wandb_logger_kwargs.name=${exp_name} \
    +exp_manager.explicit_log_dir="${result_dir}" \
    trainer.gpus=${gpus} \
    model.train_ds.ds_item=/data/train_bert_tarred_10000 \
    model.validation_ds.ds_item=[/data/IWSLT_tst2019,/data/europarl_dev,/data/news_commentary_dev,/data/news_crawl_dev,/data/rapid_dev]
set +e +x
EOF

ngc batch run \
  --instance "dgx1v.16g.${gpus}.norm" \
  --name "ml-model.bert ${exp_name}" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 94588:/data \
  --commandline "${command}"