from datasets import load_dataset
import numpy as np
import random
ds = load_dataset("1231czx/llama3_sft_w2r125k_r2r60k_r60k_ep3_tmp10", split='train')#.select(range(10000))
output_dir = ""

import re
def process_data(example):
    #txt = example['my_solu'][0].split("<|eot_id|><|start_header_id|>user<|end_header_id|>")[0]
    split_text = example['my_solu'][0].split("<|eot_id|><|start_header_id|>assistant")
    if len(split_text) > 2:
        txt = split_text[0] + "<|eot_id|><|start_header_id|>assistant" + split_text[1] + "<|eot_id|><|start_header_id|>assistant\n\n"
    if '(Yes or No)? Yes' in example['my_solu'][0]:
        label = True
    elif '(Yes or No)? No' in example['my_solu'][0]:
        label = False
    else:
        label = None
    '''
    if random.random() < example['rewards'][0]:
        txt = txt + ' \n\nIs my most recent final answer correct (Yes or No)? Yes.'
        label = True
        txt = txt + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSince your most recent response is self-evaluated as correct, no further modification is required. Thanks." + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        txt = txt + ' \n\nIs my most recent final answer correct (Yes or No)? No.'
        label = False
        txt = txt + f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSince your initial response is self-evaluated as incorrect, there might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Put your final answer within \\boxed{{}}." + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    '''
    return {"my_prompt": txt, 'proxy_reward': label}

ds = ds.map(process_data)
print(ds[1])
ds.push_to_hub(output_dir)

