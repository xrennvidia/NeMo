#!/bin/bash


python speech_to_text_ctc_bpe_ts.py \
model.tokenizer.type=bpe \
model.tokenizer.dir=/home/yangzhang/code/ts_asr/tokenizer_ls/tokenizer_spe_unigram_v128/ \
model.train_ds.manifest_filepath=/home/yangzhang/code/ts_asr/train_clean_100.json \
model.train_ds.augmentor.rir_noise_speaker.rir_manifest_path=/mnt/data/rir_noises/rir_tarred/tarred/tarred_audio_manifest.json \
model.train_ds.augmentor.rir_noise_speaker.rir_tar_filepaths=/mnt/data/rir_noises/rir_tarred/tarred/audio_{0..127}.tar \
model.train_ds.augmentor.rir_noise_speaker.noise_manifest_paths=[/mnt/data/rir_noises/noises_20s_tarred/tarred_audio_manifest.json] \
model.train_ds.augmentor.rir_noise_speaker.noise_tar_filepaths=[/mnt/data/rir_noises/noises_20s_tarred/audio_{0..255}.tar] \
model.train_ds.augmentor.rir_noise_speaker.bg_noise_manifest_paths=[/mnt/data/rir_noises/noises_20s_tarred/tarred_audio_manifest.json] \
model.train_ds.augmentor.rir_noise_speaker.bg_noise_tar_filepaths=[/mnt/data/rir_noises/noises_20s_tarred/audio_{0..255}.tar] \
trainer.max_epochs=1 \
model.train_ds.augmentor.rir_noise_speaker.bg_min_snr_db=[20] \
model.train_ds.augmentor.rir_noise_speaker.bg_max_snr_db=[50] \
model.train_ds.augmentor.rir_noise_speaker.rir_prob=0 \
model.train_ds.augmentor.rir_noise_speaker.apply_foreground_noise=False \
model.validation_ds.manifest_filepath=/mnt/data/LibriSpeech/dev_clean_overlap_0.1.json \
model.train_ds.max_duration=20 \
model.train_ds.num_workers=0 \
model.validation_ds.num_workers=0 \
model.validation_ds.batch_size=1 \
model.test_ds.num_workers=0 \
trainer.gpus=1 \
model.encoder.d_model=256 \
model.encoder.n_heads=4 \
model.encoder.n_layers=18 \
model.spec_augment.time_masks=5 \
trainer.num_sanity_val_steps=-1 \
+nemo_checkpoint_path=/home/yangzhang/code/ts_asr/stt_en_conformer_ctc_medium_v1.0.0/stt_en_conformer_ctc_medium.nemo




# PROJECT_NAME="ls_100"
# wandb login 2640ba7ea01264a146c1d9f3f075ec53350dd2f1 && python speech_to_text_ctc_bpe_ts.py \
# model.tokenizer.type=bpe \
# model.tokenizer.dir=/home/yangzhang/code/ts_asr/tokenizer_ls/tokenizer_spe_unigram_v128/ \
# model.train_ds.manifest_filepath=/home/yangzhang/code/ts_asr/train_clean_100.json \
# model.train_ds.augmentor.rir_noise_speaker.rir_manifest_path=/mnt/data/rir_noises/rir_tarred/tarred/tarred_audio_manifest.json \
# model.train_ds.augmentor.rir_noise_speaker.rir_tar_filepaths=/mnt/data/rir_noises/rir_tarred/tarred/audio_{0..127}.tar \
# model.train_ds.augmentor.rir_noise_speaker.noise_manifest_paths=[/mnt/data/rir_noises/noises_20s_tarred/tarred_audio_manifest.json] \
# model.train_ds.augmentor.rir_noise_speaker.noise_tar_filepaths=[/mnt/data/rir_noises/noises_20s_tarred/audio_{0..255}.tar] \
# model.train_ds.augmentor.rir_noise_speaker.bg_noise_manifest_paths=[/mnt/data/rir_noises/noises_20s_tarred/tarred_audio_manifest.json] \
# model.train_ds.augmentor.rir_noise_speaker.bg_noise_tar_filepaths=[/mnt/data/rir_noises/noises_20s_tarred/audio_{0..255}.tar] \
# model.validation_ds.augmentor.rir_noise_speaker.rir_manifest_path=/mnt/data/rir_noises/rir_tarred/tarred/tarred_audio_manifest.json \
# model.validation_ds.augmentor.rir_noise_speaker.rir_tar_filepaths=/mnt/data/rir_noises/rir_tarred/tarred/audio_{0..127}.tar \
# model.validation_ds.augmentor.rir_noise_speaker.noise_manifest_paths=[/mnt/data/rir_noises/noises_20s_tarred/tarred_audio_manifest.json] \
# model.validation_ds.augmentor.rir_noise_speaker.noise_tar_filepaths=[/mnt/data/rir_noises/noises_20s_tarred/audio_{0..255}.tar] \
# model.validation_ds.augmentor.rir_noise_speaker.bg_noise_manifest_paths=[/mnt/data/rir_noises/noises_20s_tarred/tarred_audio_manifest.json] \
# model.validation_ds.augmentor.rir_noise_speaker.bg_noise_tar_filepaths=[/mnt/data/rir_noises/noises_20s_tarred/audio_{0..255}.tar] \
# trainer.max_epochs=1 \
# model.train_ds.augmentor.rir_noise_speaker.bg_min_snr_db=[20] \
# model.train_ds.augmentor.rir_noise_speaker.bg_max_snr_db=[50] \
# model.train_ds.augmentor.rir_noise_speaker.rir_prob=0 \
# model.train_ds.augmentor.rir_noise_speaker.apply_foreground_noise=False \
# model.validation_ds.manifest_filepath=/home/yangzhang/code/ts_asr/dev_clean.json \
# model.train_ds.max_duration=20 \
# model.train_ds.num_workers=0 \
# model.validation_ds.num_workers=0 \
# model.validation_ds.batch_size=1 \
# model.test_ds.num_workers=0 \
# exp_manager.exp_dir=wandb_ts_asr \
# exp_manager.create_wandb_logger=true \
# +exp_manager.wandb_logger_kwargs.project=ts_asr \
# +exp_manager.wandb_logger_kwargs.name="${PROJECT_NAME}" \
# trainer.gpus=1 \
# model.encoder.d_model=256 \
# model.encoder.n_heads=4 \
# model.encoder.n_layers=18 \
# model.spec_augment.time_masks=5 \
# trainer.num_sanity_val_steps=-1 \
# +nemo_checkpoint_path=/home/yangzhang/code/ts_asr/stt_en_conformer_ctc_medium_v1.0.0/stt_en_conformer_ctc_medium.nemo

# +nemo_checkpoint_path=/home/yangzhang/code/ts_asr/stt_en_conformer_ctc_medium_v1.0.0/stt_en_conformer_ctc_large.nemo
# trainer.num_sanity_val_steps=2 \
