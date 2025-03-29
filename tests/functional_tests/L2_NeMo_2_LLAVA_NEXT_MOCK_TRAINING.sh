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
HF_HOME=/home/TestData/ykarnati/hf_data coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/vlm/llava_next/test_llava_next_train.py \
    --devices=2 \
    --max-steps=5 \
    --tensor-model-parallel-size=2 \
    --experiment-dir=/tmp/nemo2_llava_next_results/$RUN_ID
