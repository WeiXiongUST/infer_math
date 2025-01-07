from datasets import load_dataset
import numpy as np
import random
ds = load_dataset("1231czx/llama3_sft_w2r125k_r2r60k_r60k_ep3_tmp10", split='train')#.select(range(10000))


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
ds.push_to_hub("weqweasdas/llama3_sft_w2r125k_r2r60k_r60k_ep3_tmp10_vllmexp")
'''
def filter_data_0(example):
    if len(example['messages']) < 2:
        return False
    return True    

def process_data2(example):
    self_correct = False
    ans_correct = False
    if example['messages'][-1]['role'] == 'user':
        if 'Reward score:1' in example['messages'][-1]['content'] and len(example['messages']) > 3:
            self_correct = True
        if 'Reward score:1' in example['messages'][-1]['content']:
            ans_correct = True
    return {"self_correct": self_correct, "ans_correct": ans_correct}

ds = ds.map(process_data, num_proc=8)
ds = ds.filter(filter_data_0, num_proc=8)
ds = ds.map(process_data2, num_proc=8)

def filter_data_self_correct(example):
    return example['self_correct']
def filter_data_ans_correct(example):
    return example['ans_correct']

new_ds1 = ds.filter(filter_data_self_correct)
new_ds2 = ds.filter(filter_data_ans_correct)
print(new_ds1, new_ds2)
print(new_ds1[7]['gt'], new_ds1[7]['messages'])
from collections import Counter

# 假设你的数组如下
array1 = new_ds1['idx']
array2 = new_ds2['idx']
freq1 = Counter(array1)
freq2 = Counter(array2)

# 筛选符合条件的元素
#in freq1.keys()
result = [
    element for element in list(range(7500))
    if freq1.get(element, 0) <= 3 and freq2.get(element, 0) <= 35
]
print(len(result))
#print(result)

#np.save('gsm8k.npy', result)


#print(freq1[15], freq2[15])

result2 = [
    element for element in list(range(7500))
    if freq2.get(element, 0) < 1
]
print(len(result2))
'''

'''
for example in ds:
    if example['messages'][-1]['role'] != 'user':
        print(example['my_solu'])
        break
'''

'''
def filter_data(example):
    if example['messages'][-1]['role'] == 'user':
        if 'Reward score:1' in example['messages'][-1]['content'] and len(example['messages']) > 3:
            return True
    return False

new_ds = ds.filter(filter_data)
print(new_ds[0])
'''
