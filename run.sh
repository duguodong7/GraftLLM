export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \

cd LLaMA-Factory && \

llamafactory-cli train examples/train_full/llama3_full_sft_fuse_chinese.yaml &> ../logs/llama3_full_sft_fuse_chinese.txt &
# llamafactory-cli train examples/train_full/llama3_full_sft_fuse_math.yaml &> ../logs/llama3_full_sft_fuse_math.txt &

# cd alignment-handbook && \

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3.1-8b/dpo/config_full_instruction.yaml &> ../logs/dpo_instruction.txt ;

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3.1-8b/dpo/config_full_math_ini1.yaml &> ../logs/dpo_math_ini1.txt ;

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3.1-8b/dpo/config_full_math.yaml &> ../logs/dpo_math.txt ;

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3.1-8b/dpo/config_full_code_ini1.yaml &> ../logs/dpo_code_ini1.txt &