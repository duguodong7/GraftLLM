import os, json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    return model


def load_delta(model_name_or_path, compressed_delta_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = get_llama(model_name_or_path).to(device)
    delta = torch.load(compressed_delta_path)
    model.load_state_dict(delta, strict=False)
    return tokenizer, model

def decomposition(masked_input_tensor,dim=None):
    U , S , V = torch.svd(masked_input_tensor.to(torch.float32))
    
    outlier_U , outlier_V = None, None
    
    if dim is not None:
        U , S , V = U[:, :dim],S[:dim] ,V[:, :dim]
    
    return U, S, V 


def load_model(base_model,finetuned_model,dim_attn,delta_path):
    print(base_model, finetuned_model, dim_attn, delta_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model,torch_dtype=torch.bfloat16).to(device)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model,torch_dtype=torch.bfloat16).to(device)
    
    param_dict = dict()
    for k,v in base_model.state_dict().items():
        print('k:', k)
        if "self_attn" in k or "mlp" in k:
            if ".weight" in k:
                delta = finetuned_model.state_dict()[k] - v
                dim = dim_attn
                
                if "mlp" in k:
                    dim = 1400
                else:
                    dim = 1000

                U,S,V = decomposition(delta, dim=dim)
                
                k = k.replace(".weight", "")
                
                param_dict[k + ".base"] = v
                param_dict[k + ".U"] = U.data.to(torch.bfloat16)
                param_dict[k + ".S"] = S.data.to(torch.bfloat16)
                param_dict[k + ".V"] = V.data.to(torch.bfloat16)
                print(f'{k} svd completed')
    torch.save(param_dict, delta_path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_svd', action='store_true', help='llama model to load')
    parser.add_argument('--merge', action='store_true', help='llama model to load')
    parser.add_argument('--dim', type=int, default=256, help='llama model to load')
    parser.add_argument('--delta_path', type=str, default="", help='llama model to load')
    parser.add_argument('--compressed_delta_path', type=str, default="", help='llama model to load')
    parser.add_argument('--save_path', type=str, default="", help='llama model to load')
    parser.add_argument('--fintuned_model', type=str, default="", help='llama model to load')    
    parser.add_argument('--base_model', type=str, default="", help='llama model to load')    
    args = parser.parse_args()
    
    if args.use_svd:
        base_model = args.base_model
        finetuned_model = args.fintuned_model
        dim = args.dim
        delta_path = args.delta_path
        
        load_model(base_model=base_model,finetuned_model=finetuned_model,dim_attn=dim,delta_path=delta_path)

    elif args.merge:
        model_name_or_path = args.base_model
        compressed_delta_path = args.compressed_delta_path
        save_path = args.save_path
        
        tokenizer, model = load_delta(model_name_or_path=model_name_or_path, compressed_delta_path=compressed_delta_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)