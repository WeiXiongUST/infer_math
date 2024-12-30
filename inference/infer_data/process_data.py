from datasets import load_dataset
#ds1 = load_dataset("selfcorrexp2/llama3_sft_first_wrong_prompt1_generation_merged", split='train')#.select(range(22500 * 10))
#ds2 = load_dataset("selfcorrexp2/llama3_sft_first_wrong_prompt2_generation_merged", split='train')#.select(range(22500 * 10))
#ds3 = load_dataset("selfcorrexp2/llama3_sft_first_wrong_prompt3_generation_merged", split='train')#.select(range(22500 * 10))

ds = load_dataset("mytestdpo/llama3_gsm8k1_first_wrong_generation", split='train')
from datasets import concatenate_datasets, load_dataset
#ds = concatenate_datasets([ds1, ds2, ds3])


import random

def random_true_index(lst):
    # 获取所有为 True 的索引
    true_indices = [i for i, value in enumerate(lst) if value]
    # 如果没有 True 元素，返回 None
    if not true_indices:
        return None
    # 从 True 的索引中随机选择一个
    return random.choice(true_indices)
def pre_process(example):
    idx = random_true_index(example['rewards'])
    if idx is None:
        flag = False
        example['my_solu'] = [example['prompt'] + example['answers'][0]]
        #assert 1 == 0
    else:
        flag = True
        example['my_solu'] = [example['prompt'] + example['answers'][idx]]
        #example['rewards'][idx] = False
        
    example['flag'] = flag
    return example

def fil_flag(example):
    return example['flag']
nds1 = ds.map(pre_process)
nds1 = nds1.filter(fil_flag)

import re
def process_data(example):
    text = example['my_solu'][0].split("ENDSIGNAL")[0]
    text = text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n\nReach max function call limit.", "")
    pattern = r"<\|start_header_id\|>(user|assistant)<\|end_header_id\|>(.*?)(?=<\|start_header_id\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    dialogue = []
    for role, content in matches:
        dialogue.append({"role": role, "content": content.strip().replace("<|eot_id|>", "") })
    return {"messages": dialogue}
def get_turn(example):
    return {"turn":len(example['messages'])}

def filter_data0(example):
    if len(example['messages']) not in [4]:
        return False
    return True

nds2 = nds1.map(process_data)

nds2 = nds2.map(get_turn)

nds3 = nds2.filter(filter_data0)


def get_old_format_wrong2correct(example):
    # this is for wrong to correct trajectory
    new_mes = []
    new_mes.append(example['messages'][0])
    txt_tmp1 = example['messages'][1]['content'] + " \n\nIs my most recent final answer correct (Yes or No)? No."
    new_mes.append({"role":"assistant", "content": txt_tmp1})
    txt_tmp2 = f"Since your initial response is self-evaluated as incorrect, there might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Put your final answer within \\boxed{{}}."
    new_mes.append({"role":"user", "content": txt_tmp2})
    
    #new_mes.append(example['messages'][3])
    txt_tmp3 = example['messages'][3]['content'] #+ " It seems that the previous answer was actually correct." 
    # otherwise, we just use the second cot response 
    new_mes.append({"role":"assistant", "content": txt_tmp3})
    return {"conversations": new_mes}
def get_old_format_correct(example):
    new_mes = []
    new_mes.append(example['messages'][0])
    txt_tmp1 = example['messages'][1]['content'] + " \n\nIs my most recent final answer correct (Yes or No)? Yes."
    new_mes.append({"role":"assistant", "content": txt_tmp1})
    return {"conversations": new_mes}

def get_old_format_correct2correct(example):
    new_mes = []
    new_mes.append(example['messages'][0])
    txt_tmp1 = example['messages'][1]['content'] + " \n\nIs my most recent final answer correct (Yes or No)? No."
    new_mes.append({"role":"assistant", "content": txt_tmp1})
    txt_tmp2 = f"Since your initial response is self-evaluated as incorrect, there might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Put your final answer within \\boxed{{}}."
    new_mes.append({"role":"user", "content": txt_tmp2})
    txt_tmp3 = example['messages'][3]['content'] + " It seems that the previous answer was actually correct." 
    new_mes.append({"role":"assistant", "content": txt_tmp3})
    return {"conversations": new_mes}

def get_star_data(example):
    new_mes = []
    new_mes.append(example['messages'][0])
    new_mes.append(example['messages'][1])
    y = "There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution."
    new_mes.append({"role":"user", "content": y})
    new_mes.append(example['messages'][3])
    return {"conversations": new_mes}

