bash scripts/register_server.sh HanningZhang/Llama3-w2r100k-r2r40k-r182k-2ep
sleep 300
MODEL_NAME_OR_PATH=llama3

BASE_DIR=/home/wx13/infer_math/inference
OUTPUT_BASE_DIR=tmpmodelsave/step100
DATA_NAME="math"
SPLIT="test"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1

OUTPUT_DIR="./model_test_tmp10"
for ((i=0; i<=0; i+=1))
do
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.infer_test \
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
--ports "8001" \
--ports "8002" \
--ports "8003" \
--ports "8004" \
--ports "8005" \
--ports "8006" \
--ports "8007" 
done
python ../useful_codes/merge_data_script.py --base_path ${BASE_DIR}/model_test_tmp10/llama3/math --output_dir ${OUTPUT_BASE_DIR}tmp10

OUTPUT_DIR="./model_test_tmp07"
for ((i=0; i<=0; i+=1))
do
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.infer_test \
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
--ports "8001" \
--ports "8002" \
--ports "8003" \
--ports "8004" \
--ports "8005" \
--ports "8006" \
--ports "8007" 
done

python ../useful_codes/merge_data_script.py --base_path ${BASE_DIR}/model_test_tmp07/llama3/math --output_dir ${OUTPUT_BASE_DIR}tmp07

pkill -f "python -m vllm.entrypoints.api_server"
rm -rf model_test_tmp*


