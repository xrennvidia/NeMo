import difflib
from typing import Tuple


def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]

def strip_surrounding_chars(in_str_list):
    out_str = ''
    for stri in in_str_list:
        out_str += stri.lstrip('{').rstrip('}')
    return out_str


def get_eval_score(val_true: list, val_pred: list) -> Tuple:
    # tp: gets the drug name correctly
    # fp: gets the drug name incorrectly - predicts something else; and predicts drug name when there's {none}
    # fn: misses the drug name - predicts {none} while there is a drug name; and predicts something else than the actual drug name
    # tp_overlap: partially gets the drug name with overlap
    # fp_overlap: predicts comletely something else - there isn't even an overlap; and predicts drug name when there's {none}
    # fn_overlap: misses the drug name - predicts comletely something else - there isn't even an overlap; and predicts '{none}' when there's drug name

    tp, fp, fn, tp_overlap, fp_overlap, fn_overlap = 0, 0, 0, 0, 0, 0
    for si in range(len(val_true)):
        true_val = val_true[si]
        pred_val = val_pred[si]
        if not pred_val:
            pred_val = '{none}'

        pred_vals = list(set(pred_val.split(',')))
        true_vals = list(set(true_val.split(',')))
        
        if true_vals == ['{none}']:
            if pred_vals == ['{none}'] or pred_vals == ['']:
                pass # tn += 1 -> we don't count '{none}' as an entity
            else:
                fp += len(pred_vals) # false alarm
                fp_overlap += len(pred_vals)
            continue

        for pred_vali in pred_vals:
            if pred_vali in true_vals:
                tp += 1 # hit
                tp_overlap += 1 # hit
            elif pred_vali == '{none}':
                fn += 1 # miss
                fn_overlap += 1 # misses, even partially
            else:
                fp += 1 # false alarm
                fn += 1 # misses the drug name
                for pred_valii in pred_vali.lstrip('{').rstrip('}').split(' '):
                    if pred_valii.lstrip('{').rstrip('}') in true_val.lstrip('{').rstrip('}') or\
                       true_val.lstrip('{').rstrip('}') in pred_valii.lstrip('{').rstrip('}'):
                        tp_overlap += 1 # gets partially correctly
                        break
                    else:
                        fp_overlap += 1 # false alarm, even partially
                        fn_overlap += 1 # misses the drug name, even partially

    prec = tp / (fp + tp + 0.000001)
    rec = tp / (tp + fn + 0.000001)
    prec_overlap = tp_overlap / (fp_overlap + tp_overlap + 0.000001)
    rec_overlap = tp_overlap / (tp_overlap + fn_overlap + 0.000001)
    if prec == 0 or rec == 0:
        f1 = 0.0
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    if prec_overlap == 0 or rec_overlap == 0:
        f1_overlap = 0.0
    else:
        f1_overlap = 2*(prec_overlap * rec_overlap)/(prec_overlap + rec_overlap)

    return round(prec, 3), round(rec, 3), round(f1, 3),\
        round(prec_overlap, 3), round(rec_overlap, 3), round(f1_overlap, 3)

if __name__ == "__main__":
    # val_true = ['{zofran pump}', '{zofran},{nyquil}',      '{none}', '{none}',    '{birth control}', '{birth control}',  '{birth control}', '{birth control}', '{birth control}', '{zofran}',  '{zofran}']
    # val_pred = ['{zofran}', '{zofran pump},{nyquil}', '{none}', '{aspirin}', '{contraception}', '{none}',           '{none}',          '{birth control}', '{none}',          '{azofran}', '{zofran pump}']
    val_true=['{birth control}', '{birth control}' '{birth control}' '{birth control}' '{birth control}', '{zofran}']
    val_pred=['{birth control}', '{birth control}' '{birth control}' '{none}' '{birth control}', '{none}']
    print(get_eval_score(val_true, val_pred))
