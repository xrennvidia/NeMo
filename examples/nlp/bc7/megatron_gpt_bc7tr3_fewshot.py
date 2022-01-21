from collections import defaultdict
import requests, json, random, csv

import sys, csv, os
sys.path.append('examples/nlp/bc7/utils')
from utils_bc7tr3 import get_eval_score

def run(model, num_samples, tokens_to_generate, temperature):
    if model == 'nemo_1.3b':
        url = "http://10.110.42.59:9000/megagpt"
    elif model == 'megatron_530b':
        url = "http://10.14.74.235:5000/api"
    else:
        raise('unknown model/api')

    with open('/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-fewshot_train.json') as f:
        few_json = json.load(f)

    with open('/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-fewshot_val.json') as f:
        val_json = json.load(f)

    val_true = []
    val_pred = []
    none_counter = 0
    with open('gpt_fewshot_model-' + model + '_num_samples-' + str(num_samples) +\
        '_tok2gen-' + str(tokens_to_generate) + '_temp-' + str(temperature) + '.csv', 'w') as csvfw:
        csvw = csv.writer(csvfw, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for _, dri_val in enumerate(val_json.keys()):

            val_json_dri = val_json[dri_val]
            for vali in val_json_dri:

                prompts = []
                prop_none_prompts = 0.2
                num_none_prompts = int(round(num_samples * prop_none_prompts))
                num_drug_prompts = num_samples - num_none_prompts
                for dri_tr in random.sample(list(few_json.keys()), num_drug_prompts) + ['\{none\}'] * num_none_prompts:
                    fewshot_sample = random.sample(few_json[dri_tr], 1)[0]
                    prompts.append(fewshot_sample['text'].replace('"', '').replace("'", '') + '<|DRUG|>' + fewshot_sample['drug'])
                prompts_str = '\n'.join(prompts)

                val_prompt = prompts_str + '\n' + vali['text'].replace('"', '').replace("'", '') + '<|DRUG|>'
                val_true.append(vali['drug'])

                headers = {
                    "Content-Type": "application/json; charset=UTF-8"
                }
                payload = '{"prompts":["%s"], "tokens_to_generate":%d, "temperature":%f}'%((
                    val_prompt).encode('ascii', 'ignore'), tokens_to_generate, temperature)

                response = requests.request("PUT", url, headers=headers, data=payload)
                response_str = str(response.text.encode('utf-8'))
                _response_ans = response_str[response_str.rfind('<|DRUG|>')+8:]
                response_ans = _response_ans[_response_ans.find('{'):_response_ans.find('}')+1]
                val_pred.append(response_ans)
                # print(response.text.encode('utf-8'))

                if vali['drug'] != '{none}':
                    if response_ans == '':
                        response_ans = '{none}'
                    csvw.writerow([vali['text'], vali['drug'], response_ans])

    prec, rec, f1, prec_overlap, rec_overlap, f1_overlap = get_eval_score(val_true, val_pred)
    
    return prec, rec, f1, prec_overlap, rec_overlap, f1_overlap 

if __name__ == "__main__":

    # NUM_PROMPT_SAMPLES = 10
    # TOKENS_TO_GENERATE = 5
    # TEMPERATURE = 0.1
    with open('gpt_fewshot_bc7tr3_evals.csv', 'w') as csvfw:
        csvw = csv.writer(csvfw, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvw.writerow(['model', 'NUM_PROMPT_SAMPLES', 'TOKENS_TO_GENERATE', 'TEMPERATURE',
                       'prec', 'rec', 'f1', 'prec_overlap', 'rec_overlap', 'f1_overlap'])
        hyper_params = [(5, 10, 0.1), (10, 10, 0.1), (20, 10, 0.1)] 
        models = ['megatron_530b']#, 'nemo_1.3b']
        for modeli in models:
            for hpi in hyper_params:
                NUM_PROMPT_SAMPLES, TOKENS_TO_GENERATE, TEMPERATURE = hpi
                prec, rec, f1, prec_overlap, rec_overlap, f1_overlap =\
                    run('megatron_530b', NUM_PROMPT_SAMPLES, TOKENS_TO_GENERATE, TEMPERATURE)
                csvw.writerow([modeli, NUM_PROMPT_SAMPLES, TOKENS_TO_GENERATE, TEMPERATURE,
                                        prec, rec, f1, prec_overlap, rec_overlap, f1_overlap])
        