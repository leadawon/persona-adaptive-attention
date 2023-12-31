import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
from tqdm import tqdm

from config_loader.config import extend_compatibility_for_paa_transformer
from dataset.dataset import PersonaChatDataset, get_dataloader
from evaluation.evaluation_helper import eval_distinct_avg
from model.causal_decoder.conv_gpt import cut_special_tokens
from utils.get_model_by_config import get_model_via_config
from utils.get_tokenizer import get_tokenizer
from utils.parser_helper import str2bool
from utils.save_load_model import load_model

import time
import re


def save_generated_text(result_str, ground_truth, predicted, exp_name, force_acc_auto_tau, temperature, history_texts,prefix=None ):
    assert len(ground_truth) == len(predicted), 'length must equal!'
    text = result_str + "\n" + "=" * 10 + "\n"
    for gt, pred,his in zip(ground_truth, predicted,history_texts):
        text += "History    :\n{}\n\n".format(re.sub(r' \[SEP\] ','\n',his))
        text += "GroundTruth: {}\n\n".format(gt)
        text += "Prediction : {}\n".format(pred)
        text += "=" * 10 + "\n"
    folder = "generated_text/test"
    if prefix is not None:
        folder = "generated_text/test/{}".format(prefix)
    os.makedirs(folder, exist_ok=True)
    if force_acc_auto_tau:
        filename = "{}/T={}-ACC_AUTO_TAU-{}.txt".format(folder, temperature, exp_name)
    else:
        filename = "{}/T={}-{}.txt".format(folder, temperature, exp_name)
    with open(filename, 'w') as file:
        file.write(text)


parser = argparse.ArgumentParser()
parser.add_argument('--mdir', type=str, default=None)
parser.add_argument('--model_path', type=str, default='ckpt/paa.pt')
parser.add_argument('--prefix', type=str, default=None)
parser.add_argument('--max_bz', type=int, default=32)
parser.add_argument('--min_bz', type=int, default=32)
parser.add_argument('--force_acc_auto_tau', type=str2bool, default=False)
parser.add_argument('--worker', type=int, default=0)
parser.add_argument('--temperature',type=float, default=0)
# config_path = 'baseline/config/gpt2-small.yml'
from glob import glob
args = parser.parse_args()
force_acc_auto_tau = args.force_acc_auto_tau
temperature = args.temperature
MAX_BATCH_SIZE = args.max_bz
MIN_BATCH_SIZE = args.min_bz
# MAX_BATCH_SIZE = 4
# MIN_BATCH_SIZE = 4
if MAX_BATCH_SIZE < MIN_BATCH_SIZE:
    MAX_BATCH_SIZE = MIN_BATCH_SIZE
mdir = args.mdir
result = []
model_paths = glob("{}/*/best.pt".format(mdir))
if args.model_path is not None:
    model_paths = [args.model_path]
for model_path in model_paths:
    skip = False
    for entry in result:
        if model_path in entry:
            print("Skip")
            skip = True
            break
    if skip:
        continue

    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    if force_acc_auto_tau:
        config.paa_transformer.decoder.auto_tau = 'accurate'
    if config.training.batch_size < MIN_BATCH_SIZE:
        config.training.batch_size = MIN_BATCH_SIZE
    if config.training.batch_size > MAX_BATCH_SIZE:
        config.training.batch_size = MAX_BATCH_SIZE
    config = extend_compatibility_for_paa_transformer(config)
    tokenizer = get_tokenizer(config.tokenizer.vocab)
    max_context_turns = config.dataset.max_context_turns
    test_dataset = PersonaChatDataset(config.dataset.test, tokenizer.sep_token, max_context_turns=max_context_turns)
    test_dataloader = get_dataloader(test_dataset, tokenizer, config, num_workers=args.worker, batch_size_ratio=1)
    model = get_model_via_config(config, tokenizer)
    model.to('cuda')
    load_model(model, path=model_path)
    target_texts = []
    preds_texts = []
    history_texts = []
    pbar = tqdm(test_dataloader, desc='decoding tokens')
    
    #dawoncnt = 0
    
    for data in pbar:
        pred_text = model.generate(data)
        target = data['target_input']['input_ids']
        target_text = tokenizer.batch_decode(cut_special_tokens(target, tokenizer))
        history_texts += data["persona_query"]
        preds_texts += pred_text
        target_texts += target_text
        pbar.set_postfix_str(pred_text[0])
#         for k,v in data.items():
#             print("*"*10)
#             print(k)
#             print(v)
#         #print(data["query"])
#         #print(len(data["persona"]))
#         #print(tokenizer.batch_decode(cut_special_tokens(data["persona"], tokenizer)))
#         print("data"*10)
#         print(len(pred_text))
#         print("pred_text"*10)
#         dawoncnt+= 1
#         if dawoncnt == 1:
#             break
    
    assert len(history_texts) == len(preds_texts)
#     print(history_texts[0])
#     print("his"*10)
#     print(preds_texts[0])
#     print("preds"*10)
    
    dist1, dist2, avg_dist = eval_distinct_avg(preds_texts)
    result_str = """
    dist1: {}
    dist2: {}
    avg_dist: {}
    """.format(dist1, dist2, avg_dist)
    print(result_str)
    save_generated_text(result_str, target_texts, preds_texts, config.experiment.name, force_acc_auto_tau, temperature, prefix=args.prefix, history_texts = history_texts)
    del model
