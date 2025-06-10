# mkdir -p results/humaneval
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=6,7
export PATH=./vllm/bin:$PATH
export PYTHONPATH=$PYTHONPATH:./eval_plus/evalplus
# pip install datamodel_code_generator anthropic mistralai google-generativeai -i https://pypi.tuna.tsinghua.edu.cn/simple

# MODEL_DIR=${MODEL_DIR:-"/data/username/test_instruct"}
# MODEL_DIR=${MODEL_DIR:-"/data/username/grafting/saves/llama3-8b/full/sft_code_ini2"}
MODEL_DIR=${MODEL_DIR:-"/data/username/grafting/delta_paths/fuse_sft_all2"}
TP=${TP:-2}
MODEL_TYPE='llama3' # llama3 qwen2 gemma2
OUTPUT_DIR=${OUTPUT_DIR:-"/data/username/grafting/results/fuse_sft_cpr_all2/evalplus"}
mkdir -p ${OUTPUT_DIR}
echo "EvalPlus: ${MODEL_DIR}, OUTPUT_DIR ${OUTPUT_DIR}"

python generate.py \
  --model_type ${MODEL_TYPE} \
  --model_size chat \
  --model_path ${MODEL_DIR} \
  --bs 1 \
  --temperature 0 \
  --n_samples 1 \
  --greedy \
  --root ${OUTPUT_DIR} \
  --dataset mbpp \
  --tensor-parallel-size ${TP} \
  --resume

evalplus.evaluate \
  --dataset mbpp \
  --samples ${OUTPUT_DIR}/mbpp/${MODEL_TYPE}_chat_temp_0.0 > ${OUTPUT_DIR}/raw_mbpp_results.txt

# python -m evalplus.sanitize --samples ${OUTPUT_DIR}/mbpp/${MODEL_TYPE}_chat_temp_0.0

# evalplus.evaluate \
#   --dataset mbpp \
#   --samples ${OUTPUT_DIR}/mbpp/${MODEL_TYPE}_chat_temp_0.0-sanitized > ${OUTPUT_DIR}/mbpp_results.txt
