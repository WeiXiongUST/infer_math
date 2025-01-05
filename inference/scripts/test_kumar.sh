bash scripts/register_server.sh HanningZhang/Llama3-GSM8K-Star-Plus
sleep 240
MODEL_NAME_OR_PATH=llama3

BASE_DIR=/home/wx13/infer_math/inference
OUTPUT_BASE_DIR=mytestdpo/llama3_star_plus_8b_gsm8k_kumar_baseline
DATA_NAME="math"
SPLIT="test"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1

OUTPUT_DIR="./model_test_tmp10"
for ((i=0; i<=1; i+=1))
do
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.new_infer_test_kumar \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed ${i} \
--temperature 1.0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
--horizon 2 \
--ports "8000" \
--ports "8001" 
done
#python ../useful_codes/merge_data_script.py --base_path ${BASE_DIR}/model_test_tmp10/llama3/math --output_dir ${OUTPUT_BASE_DIR}tmp10

OUTPUT_DIR="./model_test_tmp07"
for ((i=0; i<=1; i+=1))
do
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.new_infer_test_kumar \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed ${i} \
--temperature 0.7 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
--horizon 2 \
--ports "8000" \
--ports "8001" 
done

python ../useful_codes/merge_data_script.py --base_path ${BASE_DIR}/model_test_tmp07/llama3/math --output_dir ${OUTPUT_BASE_DIR}tmp07


OUTPUT_DIR="./model_test_tmp0"
for ((i=0; i<=2; i+=1))
do
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.new_infer_test_kumar \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed ${i} \
--temperature 0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
--horizon 2 \
--ports "8000" \
--ports "8001" 
done

python ../useful_codes/merge_data_script.py --base_path ${BASE_DIR}/model_test_tmp0/llama3/math --output_dir ${OUTPUT_BASE_DIR}tmp0

pkill -f "python -m vllm.entrypoints.api_server"
