#!/bin/bash

# bc7-tr3 training - 126m
EXPR_NAME='bc7tr3_p_tuning'
RESTORE_PATH='/checkpoints/nemo/126m_fp16_gpt3_final.nemo'
GPUS=4
MAX_EPOCHS=10
BATCH_SIZE=16

python3 examples/nlp/text_classification/ptune_text_classification.py \
    --config-name=ptune_text_classification_config \
    trainer.gpus=$GPUS \
    trainer.max_epochs=$MAX_EPOCHS \
    exp_manager.name=$EXPR_NAME \
    +exp_manager.create_wandb_logger=True \
    +exp_manager.wandb_logger_kwargs.project=prompt_tuning \
    +exp_manager.wandb_logger_kwargs.name=p-tuning_bc7tr3_126m-01 \
    +exp_manager.resume_if_exists=True \
    +exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    +exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    +model.language_model.nemo_file=$RESTORE_PATH \
    +model.dataset.classes="['none', 'muscle', 'headache', 'medication', 'prenatal', 'vitamins', 'birth', 'control', 'patch', 'antibiotics', 'shot', 'injection', 'baby', 'aspirin', 'flu', 'vaccine', 'pills', 'white', 'heat', 'vitamin', 'B', 'COR', 'creatine', 'pain', 'pill', 'spray', 'medicine', 'oxide', 'steroid', 'oil', 'insulin', 'nausea', 'c', 'nicotine', 'morphine', 'cream', 'shots', 'saline', 'steroids', 'nasal', 'anesthesia', 'antibiotic', 'epilepsy', 'pm', 'vaccines', 'iron', 'magnesium', 'caffeine']" \
    +model.train_ds.file_path='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-p_tuning_train.json' \
    +model.valid_ds.file_path='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-p_tuning_val.json' \
    model.train_ds.batch_size=$BATCH_SIZE \
    model.validation_ds.batch_size=$BATCH_SIZE \
    model.test_ds.batch_size=$BATCH_SIZE \
    

# bc7-tr3 eval
EXPR_NAME='bc7tr3_p_tuning'
GPUS=1
NEMO_PATH='/checkpoints/nemo/p_tuning/bc7tr3/bc7tr3_p_tuning_126m-epoch50.nemo'

python3 examples/nlp/bc7/bc7_tr3_p_tuning_eval.py \
    --config-name=ptune_text_classification_config \
    trainer.gpus=$GPUS \
    exp_manager.name=$EXPR_NAME \
    model.nemo_path=$NEMO_PATH \
    +model.dataset.classes="['none', 'muscle', 'headache', 'medication', 'prenatal', 'vitamins', 'birth', 'control', 'patch', 'antibiotics', 'shot', 'injection', 'baby', 'aspirin', 'flu', 'vaccine', 'pills', 'white', 'heat', 'vitamin', 'B', 'COR', 'creatine', 'pain', 'pill', 'spray', 'medicine', 'oxide', 'steroid', 'oil', 'insulin', 'nausea', 'c', 'nicotine', 'morphine', 'cream', 'shots', 'saline', 'steroids', 'nasal', 'anesthesia', 'antibiotic', 'epilepsy', 'pm', 'vaccines', 'iron', 'magnesium', 'caffeine']" \
    +model.valid_ds.file_path='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-p_tuning_val.json'


# bc7-tr3 training - 1.3b
EXPR_NAME='bc7tr3_p_tuning'
RESTORE_PATH='/checkpoints/nemo/gpt_1p3b_fp16.nemo'
GPUS=4
MAX_EPOCHS=10
BATCH_SIZE=8

python3 examples/nlp/text_classification/ptune_text_classification.py \
    --config-name=ptune_text_classification_config \
    trainer.gpus=$GPUS \
    trainer.max_epochs=$MAX_EPOCHS \
    exp_manager.name=$EXPR_NAME \
    +exp_manager.create_wandb_logger=True \
    +exp_manager.wandb_logger_kwargs.project=prompt_tuning \
    +exp_manager.wandb_logger_kwargs.name=p-tuning_bc7tr3_1.3b-01 \
    +exp_manager.resume_if_exists=True \
    +exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    +exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    +model.language_model.nemo_file=$RESTORE_PATH \
    +model.dataset.classes="['none', 'muscle', 'headache', 'medication', 'prenatal', 'vitamins', 'birth', 'control', 'patch', 'antibiotics', 'shot', 'injection', 'baby', 'aspirin', 'flu', 'vaccine', 'pills', 'white', 'heat', 'vitamin', 'B', 'COR', 'creatine', 'pain', 'pill', 'spray', 'medicine', 'oxide', 'steroid', 'oil', 'insulin', 'nausea', 'c', 'nicotine', 'morphine', 'cream', 'shots', 'saline', 'steroids', 'nasal', 'anesthesia', 'antibiotic', 'epilepsy', 'pm', 'vaccines', 'iron', 'magnesium', 'caffeine']" \
    +model.train_ds.file_path='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-p_tuning_train.json' \
    +model.valid_ds.file_path='/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-p_tuning_val.json' \
    model.train_ds.batch_size=$BATCH_SIZE \
    model.validation_ds.batch_size=$BATCH_SIZE \
    model.test_ds.batch_size=$BATCH_SIZE \

