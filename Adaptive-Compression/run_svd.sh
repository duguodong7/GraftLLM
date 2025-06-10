#!/bin/bash
# models=(/data/username/test_sft) # models to quantize


base_models=(/data/public/Llama-3.1-8B-Instruct)
models=(/data/username/grafting/saves/llama3-8b/full/dpo_instruction_ini1) # models to quantize
delta_paths=(/data/username/grafting/delta_paths/dpo_instruction_ini1) # path to saved delta
compressed_delta_path=(/data/username/grafting/delta_paths/dpo_instruction_ini1_compressed)
save_path=(/data/username/grafting/saves/llama3-8b/full/dpo_instruction_ini1_cpr)

# base_models=(/data/public/Llama-3.1-8B-Instruct)
# models=(/data/username/grafting/saves/llama3-8b/full/sft_chinese) # models to quantize
# delta_paths=(/data/username/grafting/delta_paths/sft_chinese) # path to saved delta
# compressed_delta_path=(/data/username/grafting/delta_paths/sft_chinese_compressed)
# save_path=(/data/username/grafting/delta_paths/sft_chinese_come)

i=0

CUDA_VISIBLE_DEVICES=1 python3 load_delta.py --use_svd \
    --fintuned_model ${models[$i]} \
    --base_model ${base_models[$i]} \
    --delta_path ${delta_paths[$i]} \
    --dim 1000 \


# CUDA_VISIBLE_DEVICES=1 python3 load_delta.py --merge \
#     --base_model ${models[$i]} \
#     --compressed_delta_path ${compressed_delta_path[$i]} \
#     --save_path ${save_path[$i]}  

# CUDA_VISIBLE_DEVICES=4 python3 debug.py --merge \
#     --fintuned_model ${models[$i]} \
#     --compressed_delta_path ${compressed_delta_path[$i]} \
#     --save_path ${save_path[$i]}  

