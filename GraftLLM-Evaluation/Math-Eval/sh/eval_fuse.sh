set -ex
export CUDA_VISIBLE_DEVICES=6,7
PROMPT_TYPE="llama3-math-cot"
# MODEL_NAME_OR_PATH='/data/username/fusechat-SFT'
# OUTPUT_DIR='/data/username/grafting/FuseChat3/results_fusechat-SFT/math'

MODEL_NAME_OR_PATH='/data/username/grafting/saves/llama3-8b/full/dpo_instruction_ini1_cpr'
# MODEL_NAME_OR_PATH="/data/username/grafting/delta_paths/fuse_sft_8_2"
# MODEL_NAME_OR_PATH='/data/public/Llama-3.1-8B-Instruct'
# MODEL_NAME_OR_PATH='/data/username/test_sft'
OUTPUT_DIR='/data/username/grafting/results/dpo_instruction_ini1_cpr/math'
mkdir -p ${OUTPUT_DIR}

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="gsm8k,math,amc23"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
