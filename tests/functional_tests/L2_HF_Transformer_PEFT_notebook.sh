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
jupyter nbconvert --to script tutorials/llm/automodel/peft.ipynb --output _peft
sed -i "s#meta-llama/Llama-3.2-1B#/home/TestData/akoumparouli/hf_mixtral_2l/#g" tutorials/llm/automodel/_peft.py
sed -i "s/max_steps = 100/max_steps = 10/g" tutorials/llm/automodel/_peft.py
cp tutorials/llm/automodel/_peft.py /tmp/_peft.py
grep -iv push_to_hub /tmp/_peft.py >tutorials/llm/automodel/_peft.py
TRANSFORMERS_OFFLINE=1 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo3 tutorials/llm/automodel/_peft.py
