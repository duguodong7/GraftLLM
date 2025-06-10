MODEL_DIR='/data/username/fusechat-SFT/'
MODEL_NAME="FuseChat-Llama-3.1-8B-SFT"
NUM_GPUS=8
NUM_GPUS_PER=1

python gen_model_answer.py \
--model-path ${MODEL_DIR} \
--model-id ${MODEL_NAME} \
--num-gpus-per-model ${NUM_GPUS_PER} \
--num-gpus-total ${NUM_GPUS} \
--answer-file "data/model_answer/${MODEL_NAME}.jsonl"
