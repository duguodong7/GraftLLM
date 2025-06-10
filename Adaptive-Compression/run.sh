export PYTHONUNBUFFERED=1

attn_fp16_cols=(0) # number of fp16 columns 
mlp_fp16_cols=(0) # number of fp16 columns
attn_int8_cols=(20)
mlp_int8_cols=(20) 
attn_int4_cols=(180)  
mlp_int4_cols=(180) 
attn_int3_cols=(0)
mlp_int3_cols=(0)
attn_int2_cols=(800)
mlp_int2_cols=(1200)
bits=("8 4 2") # bits for quantization, if use int8, int3,int2, use "8 3 2"

# ("8 2"), 16,16, 984,1384

base_models=(/data/public/Llama-3.1-8B-Instruct)
# models=(/data/username/grafting/saves/llama3-8b/full/dpo_instruction_ini1) # models to quantize
# delta_paths=(/data/username/grafting/delta_paths/dpo_instruction_ini1) # path to saved delta
# compressed_delta_path=(/data/username/grafting/delta_paths/dpo_instruction_ini1_compressed)
# save_path=(/data/username/grafting/saves/llama3-8b/full/dpo_instruction_ini1_cpr)

models=(/data/username/grafting/saves/llama3-8b/full/dpo_code) # models to quantize
delta_paths=(/data/username/grafting/delta_paths/dpo_code) # path to saved delta
compressed_delta_path=(/data/username/grafting/delta_paths/dpo_code_compressed_2333)
save_path=(/data/username/grafting/saves/llama3-8b/full/dpo_code_cpr_3333)

i=0

CUDA_VISIBLE_DEVICES=6 python llama.py ${models[$i]} \
    c4 \
    --wbits 4 \
    --true-sequential \
    --act-order \
    --groupsize 128 \
    --saved_delta_path ${delta_paths[$i]} \
    --save_compressed_delta_dir ${compressed_delta_path[$i]} \
    --attn_fp16_col ${attn_fp16_cols[$i]} \
    --mlp_fp16_col ${mlp_fp16_cols[$i]} \
    --attn_int8_col ${attn_int8_cols[$i]} \
    --mlp_int8_col ${mlp_int8_cols[$i]} \
    --attn_int4_col ${attn_int4_cols[$i]} \
    --mlp_int4_col ${mlp_int4_cols[$i]} \
    --attn_int3_col ${attn_int3_cols[$i]} \
    --mlp_int3_col ${mlp_int3_cols[$i]} \
    --attn_int2_col ${attn_int2_cols[$i]} \
    --mlp_int2_col ${mlp_int2_cols[$i]} \
    --bits ${bits[$i]} 

CUDA_VISIBLE_DEVICES=6 python3 load_delta.py --merge \
    --base_model ${models[$i]} \
    --compressed_delta_path ${compressed_delta_path[$i]} \
    --save_path ${save_path[$i]}  


