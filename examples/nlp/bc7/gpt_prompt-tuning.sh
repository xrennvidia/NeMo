#!/bin/bash

# winogrande training
EXPR_NAME='winogrande_prompt_tuning'
RESTORE_PATH='/checkpoints/nemo/126m_fp16_gpt3_final.nemo'
GPUS=4
MAX_STEPS=4800
PROMPT_LENGTH=20

python3 examples/nlp/language_modeling/megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.gpus=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        restore_from_path=$RESTORE_PATH \
        exp_manager.name=$EXPR_NAME \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=[] \
        +model.new_prompt_tags=['Winogrande'] \
        +model.new_prompt_init_text=['disambiguate pronoun noun names pick correct name fill blank'] \
        +model.new_prompt_init_methods=['text'] \
        model.data.data_prefix=None \
        +model.data.train_ds='/datasets/rte_winogrande/rte_winogrande_boolq_prompt_tuned_gpt/prompt_tuning_data/winogrande_prompt_tuning_train.jsonl' \
        +model.data.valid_ds='/datasets/rte_winogrande/rte_winogrande_boolq_prompt_tuned_gpt/prompt_tuning_data/winogrande_prompt_tuning_val.jsonl' \
        +model.data.batch_size=64 \
        model.optim.lr=2e-3 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=320 \
        model.optim.sched.constant_steps=2240 \
        model.encoder_seq_length=2048 \

# winogrande eval
python3 examples/nlp/language_modeling/megatron_gpt_eval.py \
  --use_soft_prompts \
  --model_file=/checkpoints/nemo/prompt_tuning/winogrande/winogrande_prompt_tuning-bs64.nemo \
  --path_to_file=/datasets/rte_winogrande/rte_winogrande_boolq_prompt_tuned_gpt/prompt_tuning_data/winogrande_prompt_tuning_val_for_eval.jsonl \
  --tokens_to_generate=32 \
  --batch_size=16 \


# bc7-tr3 training
EXPR_NAME='bc7tr3_prompt_tuning'
RESTORE_PATH='/checkpoints/nemo/126m_fp16_gpt3_final.nemo'
GPUS=4
MAX_STEPS=48000
PROMPT_LENGTH=30
#
python3 examples/nlp/language_modeling/megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.gpus=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        restore_from_path=$RESTORE_PATH \
        exp_manager.name=$EXPR_NAME \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=[] \
        +model.new_prompt_tags=['bc7tr3-ner'] \
        +model.new_prompt_init_text=['identify drug name from tweet correct with standard name if necessary'] \
        +model.new_prompt_init_methods=['text'] \
        model.data.data_prefix=None \
        +model.data.train_ds='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-prompt_tuning_train.json' \
        +model.data.valid_ds='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-prompt_tuning_val.json' \
        +model.data.batch_size=32 \
        model.optim.lr=2e-3 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=320 \
        model.optim.sched.constant_steps=2240 \
        model.encoder_seq_length=2048 \

# bc7-tr3 eval
python3 examples/nlp/language_modeling/megatron_gpt_eval.py \
  --use_soft_prompts \
  --model_file=/checkpoints/nemo/prompt_tuning/bc7tr3/bc7tr3_prompt_tuning-loss_on_answers_only_1.3b.nemo \
  --path_to_file=/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-prompt_tuning_val.json \
  --tokens_to_generate=10 \
  --batch_size=32 \

# bc7-tr3 training - calc_loss_on_answers_only
EXPR_NAME='bc7tr3_prompt_tuning-loss_on_answers_only_5b'
RESTORE_PATH='/checkpoints/nemo/gpt_5b_fp16.nemo'
GPUS=4
MAX_STEPS=48000
PROMPT_LENGTH=100
#
python3 examples/nlp/language_modeling/megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.gpus=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        restore_from_path=$RESTORE_PATH \
        exp_manager.name=$EXPR_NAME \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=[] \
        +model.new_prompt_tags=['bc7tr3-ner'] \
        +model.new_prompt_init_text=['identify drug name from tweet correct with standard name if necessary'] \
        +model.new_prompt_init_methods=['text'] \
        model.data.data_prefix=None \
        +model.data.train_ds='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-prompt_tuning_train-loss_on_answer_True.json' \
        +model.data.valid_ds='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-prompt_tuning_val-loss_on_answer_True.json' \
        +model.calc_loss_on_answer_only=True \
        +model.data.batch_size=4 \
        model.optim.lr=2e-3 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=320 \
        model.optim.sched.constant_steps=2240 \
        model.encoder_seq_length=2048 \
