import json
import random
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser

"""
If we use multiple VLLM processes to accelerate the generation, we need to use this script to merge them.
"""


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    base_path: Optional[str] = field(
        default="./data/gen_data",
        metadata={"help": "the location dir of the output file"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the huggingface address of the dataset"},
    )
    num_datasets: Optional[int] = field(
        default=8,
        metadata={"help": "the location of the output file"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


all_dirs = [script_args.base_path + str(i) + ".json" for i in range(script_args.num_datasets)]

gathered_data = []
for my_dir in all_dirs:
    ds = load_dataset("json", data_files=my_dir, split="train")
    print(len(ds))
    for sample in ds:
        gathered_data.append(sample)

random.shuffle(gathered_data)

all_data = gathered_data

output_dir = script_args.output_dir


keys = all_data[0].keys()  

dict_data = {key: [d[key] for d in all_data] for key in keys}

from datasets import Dataset, DatasetDict 
dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub(output_dir)


