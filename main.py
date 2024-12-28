import os
import json
import torch
import argparse
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(tar_mod_nam):
    device = "cuda"
    tar_mod = AutoModelForCausalLM.from_pretrained(tar_mod_nam, return_dict=True).to(device)
    tar_mod.eval()
    tar_tok = AutoTokenizer.from_pretrained(tar_mod_nam)

    return tar_mod, tar_tok


def cal_ppl_for_all_token(text, model, tok):
    prob_dis = []
    device = model.device
    input_ids = tok.encode(text)
    input_ids_list = [[tok.bos_token_id] + input_ids[-i:] for i in range(1, len(input_ids)+1)]

    for input_ids in input_ids_list:
        input_ids = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)

        logit = output[1]
        # 第一个 token 输入之后输出的是第二个 token 的概率分布
        probs = torch.nn.functional.softmax(logit, dim=-1)[0][:-1, :]
        # 第 win_siz 个 token 的 porb 跟原句子的推理一样，都只是基于 win_siz个 token
        probs = probs[range(len(probs)), input_ids[0][1:]].tolist()
        prob_dis.append(probs)

    input_ids = input_ids[0].tolist()
    print(input_ids)
    print(prob_dis)

    return prob_dis, input_ids


def cal_ppl_lim_pre(text, model, tok, pre_len):
    device = model.device
    input_ids = tok.encode(text)
    input_ids_list = [input_ids[i:i+pre_len+1] for i in range(len(input_ids)-pre_len)]

    input_ids_list = torch.tensor(input_ids_list).to(device)
    # logit = []
    # for i in range(0, len(input_ids_list), 150):
    with torch.no_grad():
        output = model(input_ids_list, labels=input_ids_list)

        # logit.extend(output[1].tolist())
    # 第一个 token 输入之后输出的是第二个 token 的概率分布
    logit = output[1]
    input_ids_list = input_ids_list[:, 1:]
    input_ids_list = input_ids_list.reshape(-1)
    prob_dis = torch.nn.functional.softmax(logit, dim=-1)[:, :-1, :].reshape(len(input_ids_list), -1)
    # 第 win_siz 个 token 的 porb 跟原句子的推理一样，都只是基于 win_siz个 token
    prob_dis = prob_dis[range(len(input_ids_list)), input_ids_list[:]]
    prob_dis = prob_dis.reshape(-1, pre_len).detach().cpu()

    return prob_dis, input_ids


def cal_ppl(text, model, tok):
    device = model.device
    input_ids = tok.encode(text)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = model(input_ids, labels=input_ids)

    logit = output[1]

    # Apply softmax to the logits to get probabilities
    prob = torch.nn.functional.softmax(logit, dim=-1)[0][:-1]
    input_ids = input_ids[0][1:]

    probs = prob[torch.arange(len(prob)), input_ids].detach().cpu()
    input_ids = input_ids.tolist()
    return probs, input_ids


def inference(text, label, tar_mod, tar_tok, pre_len):
    response = {}
    if pre_len == -1:
        prob_dis, input_ids = cal_ppl(text, tar_mod, tar_tok)
    else:
        prob_dis, input_ids = cal_ppl_lim_pre(text, tar_mod, tar_tok, pre_len)

    response["input_ids"] = input_ids
    response["prob_dis"] = prob_dis
    response["text"] = text
    response["label"] = label
    return response


def gen_pro_dis(dat, key_nam, tar_mod, tar_tok, pre_len):
    responses = []
    for example in tqdm(dat):
        text = example[key_nam]
        label = example["label"]
        response = inference(text, label, tar_mod, tar_tok, pre_len)
        responses.append(response)

    return responses


data = "wikipedia_ngram_7_0.2"
tar_mod = "pythia-6-9B"
key_nam = "text"
print(f"detect {data} from {tar_mod}.")

out_dir = "output"
out_pat = os.path.join(out_dir, data)
Path(out_pat).mkdir(parents=True, exist_ok=True)

dat_dir = "../datasets/wikipedia"
dat_pat = os.path.join(dat_dir, f"{data}.jsonl")
with open(dat_pat, 'r') as f:
    dataset = [json.loads(line) for line in f]

mod_dir = "../models"
tar_mod_name = os.path.join(mod_dir, tar_mod)
tar_model, tar_tokenizer = load_model(tar_mod_name)

# 基于全部上下文的词元概率计算 （pre_len = -1 时）
pre_len = -1
pro_dis = gen_pro_dis(dataset, key_nam, tar_model, tar_tokenizer, pre_len)
with open(f"{out_pat}/{tar_mod}.pkl", "wb") as f:
    pkl.dump(pro_dis, f)

# 基于短距离上下文的次元概率计算
pre_lens = [1,2,3,4,5,6,7]
for pre_len in pre_lens:
    print(f"detect {data} from {tar_mod}.")
    print(f"generate probability distribution conditioned on prefix with length {pre_len}.")

    out_dir = "output"
    out_pat = os.path.join(out_dir, data)
    Path(out_pat).mkdir(parents=True, exist_ok=True)

    dat_dir = "../datasets/wikipedia"
    dat_pat = os.path.join(dat_dir, f"{data}.jsonl")
    with open(dat_pat, 'r') as f:
        dataset = [json.loads(line) for line in f]

    mod_dir = "../models"
    tar_mod_name = os.path.join(mod_dir, tar_mod)
    tar_model, tar_tokenizer = load_model(tar_mod_name)
    pro_dis = gen_pro_dis(dataset, key_nam, tar_model, tar_tokenizer, pre_len)

    with open(f"{out_pat}/{tar_mod}-{pre_len}.pkl", "wb") as f:
        pkl.dump(pro_dis, f)

