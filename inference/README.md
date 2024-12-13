## Inference for MATH Problem with External Python Intepreter 

In this repo, we implement the MATH problem solving with external python tool using the VLLM to accelerate infernece. The codebase is largely based on the Tora project.


## 1 Installation instructions

**Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!**


Before starting, please make sure your linux machine has nvidia-cuda-toolkit installed. See SFT part for the guidance. 


```sh
conda create -n infer python=3.10.9
conda activate infer
# You may check nvcc -V , if no nvcc exists, you may run the following code
# conda install nvidia/label/cuda-12.2.0::cuda-nvcc

pip install datasets

# The following code is tested for CUDA12.0-12.2 and CUDA12.6
# To develop llama-3, mistral, gemma-1, 1.1, 2, deepseek you can consider the following vllm version
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!


huggingface-cli login
pip install pebble
pip install timeout_decorator
pip install ipython
pip install sympy==1.12
pip install antlr4-python3-runtime==4.11 # The versions of sympy and antlr4 cannot be modified!!!!!
```


## 2 Running the Generation Code
You can change the model, prompt set and genration config in gen_cot.sh. Only the problem, gt, and index_level_0 (index) in selfrew/augment_math will be used.

```sh
bash scripts/gen_cot.sh 
```

Then, you can use merge_data.py to merge them into a huggingface dataset.

## 4 Annotate Data

To annotate data, you need to process the generated responses in to a list (of length 2), where each component is one round of generation. For instance, sample['codes'] = [solution of 1st round, solution of 2nd round] and the script will return a list of rewards: [True, False], meaning that the model is correct in the first round but is incorrect in the second round.
    
```sh
python -um infer_data.annotate_data
```

You can consider split the generated response by ``Is my most recent final answer correct''. I provide an example of SFT data here: selfrew/filtered_data_sft.
