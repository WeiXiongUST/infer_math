import json
import random
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser
# This script finds all the files ending with jsonl and merge them into one file 
import os
from datasets import load_dataset, Dataset, DatasetDict
import json

"""
If we use multiple VLLM processes to accelerate the generation, we need to use this script to merge them.
"""


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    base_path: Optional[str] = field(
        default="",
        metadata={"help": "the location dir of the output file"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the huggingface address of the dataset"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

jsonl_files = [script_args.base_path + '/' + f for f in os.listdir(script_args.base_path) if f.endswith('.jsonl')]

all_data = []
for dir_ in jsonl_files:
    ds_test = load_dataset('json', data_files=dir_, split='train')
    for sample in ds_test:
        all_data.append(sample)


output_dir = script_args.output_dir


keys = all_data[0].keys()  

dict_data = {key: [d[key] for d in all_data] for key in keys}

dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub(output_dir)


