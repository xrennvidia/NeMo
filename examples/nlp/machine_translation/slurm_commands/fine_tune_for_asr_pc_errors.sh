#!/bin/bash
#SBATCH -A ent_aiapps_asr
#SBATCH -p batch_dgx2h_m2                 # luna / backfill / interactive
#SBATCH -N 1                    # number of nodes
#SBATCH -t 8:00:00              # wall time  (4 for luna, 8 for backfill, 2 for interactive)
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus-per-node=16
#SBATCH -J "ent_aiapps_asr:punctuation_capitalization_using_NMT_large24x6_bs18000"  # job name (<< CHANGE ! >>)
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --overcommit
#SBATCH --ntasks-per-node=16     # n tasks per machine (one task per gpu) <required>
set -x
USERID='apeganov'
CONTAINER="gitlab-master.nvidia.com/apeganov/speechtranslation:latest"
WANDB="${wandb}" # replace with your own WandB API key

# Training - we want to train for 300B tokens with a global batch size of at least 1M tokens
# total_tokens = max_steps * global_batch_size_in_tokens
# global_batch_size_in_tokens = micro_batch_size * data_parallel_size * accumulate_grad_batches * seq_length
# data_parallel_size = num_nodes * num_gpus_per_node (no model parallel)
MAX_STEPS=150000
VAL_CHECK_INTERVAL=500
LOG_EVERY_N_STEPS=100
LR=2e-5
WARMUP_STEPS=2000

# Logging
PROJECT="autoregressive_punctuation_capitalization"
EXPNAME="fine_tune_NMT_for_ASR_PC_errors"

# Mounts
SLURM_ACCOUNT='ent_aiapps'
USERID='apeganov'
LUSTRE_ACCOUNT_PREFIX=/gpfs/fs1/projects/${SLURM_ACCOUNT}
DATA="${LUSTRE_ACCOUNT_PREFIX}/datasets/data/punctuation_capitalization/ASR_PC_augmentation_of_en_WMT_src"
RESULTS=${LUSTRE_ACCOUNT_PREFIX}/users/${USERID}/results/$PROJECT/$EXPNAME
CODE="${LUSTRE_ACCOUNT_PREFIX}/users/${USERID}/code/NeMo"

mkdir -p ${RESULTS}

MOUNTS="--container-mounts=$CODE:/code,$RESULTS:/results,$DATA:/data"

# Necessary Exports
export HYDRA_FULL_ERROR=1

OUTFILE="${RESULTS}/slurm-%j-%n.out"
ERRFILE="${RESULTS}/error-%j-%n.out"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB} \
&& echo "Starting training" \
&& cd /code/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python \
  /code/examples/nlp/machine_translation/enc_dec_nmt.py \
	--config-path=/code/examples/nlp/machine_translation/conf \
	--config-name=aayn_finetuning_for_asr_2_val_sets_big_data \
	model.train_ds.use_tarred_dataset=true \
	nemo_file=/gpfs/fs1/projects/ent_aiapps/users/apeganov/pretrained_checkpoints/wmt21_en_de_backtranslated_24x6_averaged.nemo \
	model.train_ds.metadata_file="/data/train_tarred_10000/metadata.tokens.10000.json" \
	model.validation_ds.src_file_name=[/data/IWSLT_tst2019_T5_evelina_mwer/en,/data/IWSLT_tst2019_TTS_ASR_augmented/en] \
	model.validation_ds.tgt_file_name=[/data/IWSLT_tst2019_T5_evelina_mwer/de,/data/IWSLT_tst2019_TTS_ASR_augmented/de] \
	model.test_ds.src_file_name=[/data/IWSLT_tst2019_T5_evelina_mwer/en,/data/IWSLT_tst2019_TTS_ASR_augmented/en] \
	model.test_ds.tgt_file_name=[/data/IWSLT_tst2019_T5_evelina_mwer/de,/data/IWSLT_tst2019_TTS_ASR_augmented/de] \
	model.encoder_tokenizer.tokenizer_model="/data/shared_tokenizer.32000.BPE.model" \
	model.decoder_tokenizer.tokenizer_model="/data/shared_tokenizer.32000.BPE.model" \
	model.optim.lr="${LR}" \
	~model.optim.sched.warmup_ratio \
	+model.optim.sched.warmup_steps="${WARMUP_STEPS}" \
	trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
	trainer.gpus=${SLURM_NTASKS_PER_NODE} \
	trainer.max_steps=${MAX_STEPS} \
	trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
	exp_manager.create_wandb_logger=true \
	exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
	exp_manager.wandb_logger_kwargs.project=${PROJECT} \
	+exp_manager.explicit_log_dir=/results \
	+exp_manager.resume_if_exists=True \
	+exp_manager.resume_ignore_no_checkpoint=True \
	exp_manager.create_checkpoint_callback=True \
	exp_manager.checkpoint_callback_params.monitor=val_CER \
	+exp_manager.checkpoint_callback_params.save_top_k=3 \
	exp_manager.checkpoint_callback_params.mode=min \
	+exp_manager.checkpoint_callback_params.always_save_nemo=False
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
set +x
