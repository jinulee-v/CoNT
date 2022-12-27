"""
APT scoring evaluation
"""
import argparse
import glob
import os
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.utils.rnn import pad_sequence

hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
device = torch.device("cpu")
batch_size=32
pad_id = tokenizer.pad_token_id

def load_jsonl(path):
    inst_list = []
    with open(path) as f:
        for line in f:
            inst_list.append(json.loads(line))
    return inst_list

def get_mi_score(s1, s2):
    assert len(s1) == len(s2)
    length = len(s1)
    tokenized_input_seq_pair = tokenizer.batch_encode_plus(list(zip(s1, s2)), max_length=256, return_token_type_ids=True, truncation=True)
    input_ids = [torch.tensor(x) for x in tokenized_input_seq_pair["input_ids"]]
    token_type_ids = [torch.tensor(x) for x in tokenized_input_seq_pair["token_type_ids"]]
    attention_masks = [torch.tensor(x) for x in tokenized_input_seq_pair["attention_mask"]]
    predicted_probability = None
    print("RoBERTa_large for NLI judgement...")
    for head in tqdm(range(0, length - 1, batch_size)):
        tail = min(head + batch_size, length)
        with torch.no_grad():
            input_id = pad_sequence(input_ids[head:tail], batch_first=True, padding_value=pad_id).to(device)
            token_type_id = pad_sequence(token_type_ids[head:tail], batch_first=True, padding_value=0).to(device)
            attention_mask = pad_sequence(attention_masks[head:tail], batch_first=True, padding_value=0).to(device)
            outputs = model(
                input_id,
                attention_mask=attention_mask,
                token_type_ids=token_type_id,
                labels=None,
            )
            if predicted_probability is None:
                predicted_probability = torch.softmax(outputs[0], dim=1)
            else:
                predicted_probability = torch.cat((predicted_probability, torch.softmax(outputs[0], dim=1)), dim=0)
    return predicted_probability

def get_apt_score(s1, s2):
    """
    """
    mi_score = (torch.argmax(get_mi_score(s1, s2), dim=1) == 0) * (torch.argmax(get_mi_score(s2, s1), dim=1) == 0)
    mi_score = mi_score.float()

    from bleurt.score import BleurtScorer
    bleurt_scorer = BleurtScorer()
    bleurt_score = torch.tensor([(x+y)/2 for x, y in zip(bleurt_scorer.score(references=s1, candidates=s2), bleurt_scorer.score(references=s2, candidates=s1))]).to(device)

    return mi_score / ((1 + torch.exp(5 * bleurt_score)) ** 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys_file', default=None, type=str)
    parser.add_argument('--sys_path', default=None, type=str)
    parser.add_argument('--gpus', default=None, type=str)
    args = parser.parse_args()
    if args.sys_path is not None:
        candidate_files = glob.glob(os.path.join(args.sys_path, "*.sys"))
    else:
        candidate_files = [args.sys_file]
    if args.gpus:
        gpu_id = args.gpus.split(',')[0] # Only single core
        device = torch.device("cuda:"+gpu_id)
        model = model.to(device)

    for cand_file in candidate_files:
        ref_path = cand_file.replace('sys', 'ref')
        sys_path = cand_file
        print("evaluate: ", sys_path)
        sys_outputs = load_jsonl(sys_path)
        sys_outputs = [x["sys_out"] for x in sys_outputs]
        ref_outputs = load_jsonl(ref_path)
        ref_outputs = [x["ref_out"] for x in ref_outputs]
        assert len(sys_outputs) == len(ref_outputs)
        
        # Calculate apt score
        apt_score = get_apt_score(sys_outputs, ref_outputs).tolist()
        
        # Statistics
        apt_score = sorted(apt_score)
        print("mean: ", sum(apt_score) / len(apt_score))
        print("Q1:   ", apt_score[1 * len(apt_score) // 4])
        print("Q2:   ", apt_score[2 * len(apt_score) // 4])
        print("Q3:   ", apt_score[3 * len(apt_score) // 4])

