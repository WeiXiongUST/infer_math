"""
This scrip support is adapted from Tora project and supports multi-rounds vllm inference.
The inference is formulated as a multi-turn chat and the model should be registered as a server by scripts/register_server.sh first.
"""
from datasets import load_dataset, Dataset, DatasetDict

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from datasets import load_dataset
import requests
from eval.evaluate import evaluate
from eval.evaluate import get_batch_scores
from tqdm import tqdm
from utils.data_loader import load_data
from utils.parser import *
from utils.python_executor import PythonExecutor
from utils.utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from vllm import LLM, SamplingParams


import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="uf_split0_responses_K8.jsonl",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the output file"},
    )




parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ds = load_dataset(script_args.dataset_name_or_path, split="train")

remain_codes = []
remain_gts = []
all_samples = []
for sample in ds:
    remain_codes.append(sample['codes'])
    remain_gts.append(sample['gt'])
    all_samples.append(sample)
all_rm_scores = get_batch_scores(remain_codes, remain_gts)

all_data = []

for i, sample in enumerate(all_samples):
    sample.update({"rewards": all_rm_scores[i]})
    all_data.append(sample)

keys = all_data[0].keys()  

dict_data = {key: [d[key] for d in all_data] for key in keys}


dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub(output_dir)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
