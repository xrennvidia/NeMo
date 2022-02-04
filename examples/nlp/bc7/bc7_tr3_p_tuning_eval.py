import os
import pathlib
from tabnanny import check

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.text_classification.ptune_text_classification_model import (
    PTuneTextClassificationModel,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

import json, csv, sys
sys.path.append('examples/nlp/bc7/utils')
from utils_bc7tr3 import get_eval_score


@hydra_runner(config_path="../text_classification/conf", config_name="ptune_text_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'\nConfig Params:\n{OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **cfg.trainer)
    checkpoint_path = cfg.model.nemo_path
    eval_model = PTuneTextClassificationModel.restore_from(
            restore_path=checkpoint_path, trainer=trainer)

    prompts = open(cfg.model.valid_ds.file_path, 'r')

    answers = []
    prompts_org = []

    none_counter = 0
    for _prompt in prompts.readlines():
        prompt = json.loads(_prompt)
        sentence = prompt['sentence']
        answer = prompt['label']

        if answer == 'none':
            none_counter += 1
            if none_counter > 100:
                continue
                
        prompts_org.append(sentence)
        answers.append('{' + answer + '}')

    _predicts = eval_model.cuda().classifytext(
            queries=prompts_org, batch_size=8, prompt='Drug'
        )
    predicts = []
    for i, predi in enumerate(_predicts):
        predicts.append('{' + predi + '}')

    ckpt_fname = os.path.basename(checkpoint_path[:-5])

    with open('gpt_prompt_tuning-' + ckpt_fname +\
        '_indvs.csv', 'w', encoding='utf-8') as csvfw:
        csvw = csv.writer(csvfw, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i, sentence in enumerate(prompts_org):
            csvw.writerow([prompts_org[i], answers[i], predicts[i]])

    eval_score = get_eval_score(answers, predicts)
    with open('gpt_prompt_tuning-' + ckpt_fname +\
        '.csv', 'w') as csvfw:
        csvw = csv.writer(csvfw, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvw.writerow(eval_score)


if __name__ == '__main__':
    main()
