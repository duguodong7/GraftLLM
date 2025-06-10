#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
export HF_ENDPOINT=https://hf-mirror.com
global_record_file="eval_record_collection.csv"
selected_subjects="all"
gpu_util=0.95
# MODEL_NAME_OR_PATH="/data/username/grafting/saves/llama3-8b/full/sft_chinese"
# MODEL_NAME_OR_PATH="/data/username/grafting/delta_paths/sft_chinese_come"
MODEL_NAME_OR_PATH="/data/username/test_instruct"
selected_dataset="gpqa_diamond"  #mmlu, gpqa_diamond, mmlu_pro
save_dir='/data/username/grafting/results/fuse_instruct/'${selected_dataset}
mkdir -p ${save_dir}

 python evaluate_from_local.py \
--selected_subjects $selected_subjects \
--chat_template  \
--zero_shot  \
--selected_dataset $selected_dataset \
--save_dir $save_dir \
--model $MODEL_NAME_OR_PATH \
--global_record_file $global_record_file \
--gpu_util $gpu_util



