# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/speechlm/speech_to_text_llm_train.py \
  --train_manifest /home/TestData/speechlm/speechlm_data/speech_to_text_debug2/debug_2.json \
  --val_manifest /home/TestData/speechlm/speechlm_data/speech_to_text_debug2/debug_2.json \
  --restore_path /home/TestData/nemo2_ckpt/llama_68M \
  --devices 2 \
  --max_steps 500 \
  --experiment_dir /tmp/nemo2_speechlm_lora/$RUN_ID \
  --peft lora \
  --tp_size 1 \
  --pp_size 1 \
  --mbs 2

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/speechlm/speech_to_text_llm_train.py \
  --train_manifest /home/TestData/speechlm/speechlm_data/speech_to_text_debug2/debug_2.json \
  --val_manifest /home/TestData/speechlm/speechlm_data/speech_to_text_debug2/debug_2.json \
  --restore_path /home/TestData/nemo2_ckpt/llama_68M \
  --devices 2 \
  --max_steps 600 \
  --experiment_dir /tmp/nemo2_speechlm_lora/$RUN_ID \
  --peft lora \
  --tp_size 1 \
  --pp_size 1 \
  --mbs 2
