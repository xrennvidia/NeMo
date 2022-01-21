import os, json, unittest
import pandas as pd


bc7_tr3_datadir = '/datasets/bc7/Track-3_Med-Tweets'

class DataTest(unittest.TestCase):

    def test_train(self):
        org = load_tsv_convert_to_json('train', testing=True)
        rec = load_json('train')
        self.assertTrue(org.equals(rec))

    def test_val(self):
        org = load_tsv_convert_to_json('val', testing=True)
        rec = load_json('val')
        self.assertTrue(org.equals(rec))


def load_json(part: str) -> str:
    if part == 'train':
        json_file = os.path.join(bc7_tr3_datadir, 'bc7_tr3-train.json')
    elif part == 'val':
        json_file = os.path.join(bc7_tr3_datadir, 'bc7_tr3-val.json')
    else:
        raise('Unsupported partition - option: [train; val]')

    dict_list = []
    with open(json_file, 'r') as rf:
        while True:
            line = rf.readline()
            if not line:
                break
            dict_list.append(json.loads(line))

        df = pd.DataFrame(dict_list)
    return df


def load_tsv_convert_to_json(part: str = 'train', method: str = 'few-shot', calc_loss_on_answer: bool = False) -> bool:
    # part: train/val
    # method: fine-tuningfew-shot/prompt-tuning
    # returns JSON str when testing

    if part == 'train':
        tr_df_0 = pd.read_csv(os.path.join(bc7_tr3_datadir, 'BioCreative_TrainTask3.0.tsv'), sep='\t')
        tr_df_1 = pd.read_csv(os.path.join(bc7_tr3_datadir, 'BioCreative_TrainTask3.1.tsv'), sep='\t')
        df = pd.concat([tr_df_0, tr_df_1])
    elif part == 'val':
        df = pd.read_csv(os.path.join(bc7_tr3_datadir, 'BioCreative_ValTask3_corrected.tsv'), sep='\t')
    else:
        raise('Unsupported partition - option: [train; val]')

    df.loc[df['drug']=='-', 'drug'] = 'none'
    df['drug'] = '{' + df['drug'] + '}'

    df2 = df.groupby(['tweet_id', 'text'])['drug'].apply(','.join).reset_index()

    df_ft = df2.copy()
    df_ft['text'] = '<|endoftext|>' + df2['text'] + '<|drug|>' + df2['drug'] + '<|endoftext|>'
        
    df_json = df_ft[['tweet_id', 'text', 'drug']].to_json(orient='records')
    parsed = json.loads(df_json)

    if method == 'fine-tuning':
        # write fine-tuning data
        with open(os.path.join(bc7_tr3_datadir, 'bc7_tr3-' + part + '.json'), 'w', encoding='utf-8') as f:
            for jsoni in parsed:
                f.write(json.dumps(jsoni) + '\n')
    elif method == 'few-shot':
        # few-shot data
        few_shot_drug_samples_dict = {}
        drugnames_list = df[~df['drug'].str.contains("\{none\}")]['drug'].unique().tolist()
        for drni in drugnames_list:
            dfni = df2[df2['drug'].str.contains(drni)]
            few_shot_drug_samples_dict[drni.lstrip('"').rstrip('"')] = json.loads(dfni[['text','drug']].to_json(orient='records'))
        
        few_shot_drug_samples_dict['\{none\}'] = json.loads(df2[df2['drug'].str.contains('\{none\}')].to_json(orient='records'))
        with open(os.path.join(bc7_tr3_datadir, 'bc7_tr3-fewshot_' + part + '.json'), 'w', encoding='utf-8') as f:
            json.dump(few_shot_drug_samples_dict, f, indent=2)
    elif method == 'prompt-tuning':
        df2['prompt_tag'] = 'bc7tr3-ner'
        if calc_loss_on_answer:
            df2['answer'] = df2['drug']
            df3 = df2[['prompt_tag', 'text', 'answer']]
        else:
            df2['_text'] = df2['text']
            df2['text'] = df2['_text'] + ' [answer]: ' + df2['drug']
            df3 = df2[['prompt_tag', 'text']]
        prompt_tuning_dict = json.loads(df3.to_json(orient='records'))
        with open(os.path.join(bc7_tr3_datadir, 'bc7_tr3-prompt_tuning_' +\
            part + '-loss_on_answer_' + str(calc_loss_on_answer) + '.json'), 
            'w', encoding='utf-8') as f:
            for jsoni in prompt_tuning_dict:
                f.write(json.dumps(jsoni) + '\n')
    else:
        raise('unknown method')

    return True

    
if __name__ == "__main__":
    results = []
    methods = ['prompt-tuning']#'few-shot', 'prompt-tuning']
    calc_loss_on_answer = True
    for methodi in methods:
        results.append(load_tsv_convert_to_json('train', methodi, calc_loss_on_answer))
        results.append(load_tsv_convert_to_json('val', methodi, calc_loss_on_answer))
    print(all(results))
    # unittest.main()
