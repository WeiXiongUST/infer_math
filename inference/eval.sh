bash scripts/test_model.sh
rm -rf model_test_tmp*
pkill -f "python -m vllm.entrypoints.api_server"
sleep 60
bash scripts/test_model2.sh


