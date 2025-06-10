
export PYTHONUNBUFFERED=1

attn_fp16_cols=(0) # number of fp16 columns 
mlp_fp16_cols=(0) # number of fp16 columns
attn_int8_cols=(2)
mlp_int8_cols=(2) 
attn_int4_cols=(0)  
mlp_int4_cols=(0) 
attn_int3_cols=(32)
mlp_int3_cols=(32)
attn_int2_cols=(968)
mlp_int2_cols=(1428)
bits=("8 3 2") # bits for quantization, if use int8, int3,int2, use "8 3 2"
# models=(/data/username/test_sft) # models to quantize
# models=(/data/public/Llama-3.1-8B-Instruct)
# delta_paths=(/data/username/grafting/delta_paths/fuse_sft) # path to saved delta
# save_dir=(/data/username/grafting/delta_paths/fuse_sft_compressed)

models=(/data/username/grafting/saves/llama3-8b/full/sft_chinese) # models to quantize
delta_paths=(/data/username/grafting/delta_paths/sft_chinese) # path to saved delta
save_dir=(/data/username/grafting/delta_paths/sft_chinese_compressed)

i=0

CUDA_VISIBLE_DEVICES=6 python llama.py ${models[$i]} \
    c4 \
    --wbits 4 \
    --true-sequential \
    --act-order \
    --groupsize 128 \
    --saved_delta_path ${delta_paths[$i]} \
    --save_compressed_delta_dir ${save_dir[$i]} \
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



