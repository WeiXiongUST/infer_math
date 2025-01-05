## SFT Environment

```shell
conda create -n sft python=3.10.9
conda activate sft
# cuda if need
#conda install nvidia/label/cuda-12.2.0::cuda-nvcc

## Get axolotl for general model
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout 55cc214c767741e83ee7b346e5e13e6c03b7b9fa
pip install -e .

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
# you may encounter underfined symbol error related to cuda and flash-attn and 2.1.2 can solve it ...
pip3 install torch==2.1.2 torchvision torchaudio
pip install flash-attn==2.6.3

# fix an error of axolotl: ModuleNotFoundError: No module named 'pynvml.nvml'; 'pynvml' is not a package
pip install nvidia-ml-py3
# also edit axolotl/src/axolotl/utils/bench.py (line 6) to: ``from pynvml import NVMLError''


## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
pip install deepspeed==0.14.5
pip install xformers==0.0.23
pip3 install torch==2.1.2 torchvision torchaudio
pip install transformers==4.44.1
# then you need to edit /home/wx13/anaconda3/envs/sft/lib/python3.10/site-packages/transformers/integrations/deepspeed.py
# around line 245,                     0.9 * hidden_size * hidden_size -> int(0.9 * hidden_size * hidden_size)
```

You also need to install wandb to record the training and log in with the huggingface accout to access Gemma.

```shell
pip install wandb
wandb login

huggingface-cli login
```

# Hack axoltol 
update axoltol/src/axoltol/train.py, around line 72.

train_dataset.map(modify_list)


```python
def is_sublist(sublist, larger_list):
    return str(sublist)[1:-1] in str(larger_list)[1:-1]
def set_minus_100(arr):
    inside_non_minus_100 = False  # 标记是否进入非 -100 区间
    for i in range(len(arr)):
        if arr[i] != -100:
            if not inside_non_minus_100:  # 第一次进入非 -100 区间
                inside_non_minus_100 = True
            arr[i] = -100  # 修改为 -100
        elif inside_non_minus_100:  # 一旦重新遇到 -100，停止处理
            break
    return arr
        
def modify_list(example):
    lst = example['labels']
    
    sequence = [4815, 3957, 856, 1455] # \n\nIs my most ...
    #sequence = [4800, 358, 690, 8881, 389]
    def find_sequence_start(lst_tmp, sequence):
        for i in range(len(lst_tmp) - len(sequence) + 1):
            if lst_tmp[i:i+len(sequence)] == sequence:
                return i
        return -1
    start_index = find_sequence_start(lst, sequence)
    # [320, 9642, 477, 2360, 12106, 7566, 13], ' (Yes or No)? Yes.'

    #if is_sublist([320, 9642, 477, 2360, 12106, 7566, 13], lst):
        #return example
    # Update elements prior to the sequence
    if start_index != -1:
        lst[:start_index] = [-100] * start_index
    else:
        assert 1 == 0
    # [1102, 5084, 430, 279, 3766, 4320, 574, 3604, 4495, 13] " It seems that the previous answer was actually correct."

    if is_sublist([1102, 5084, 430, 279, 3766, 4320, 574, 3604, 4495, 13], example['labels']):
        lst = set_minus_100(lst)
    example['labels'] = lst
    return example


```

