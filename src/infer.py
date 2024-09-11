import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#TODO: quantization, if needed, does not work with pipeline: 
# https://stackoverflow.com/questions/70609579/use-quantization-on-huggingface-transformers-models

#TODO: leaving the inference part simple for now, but might need to look into how to speed it up
# for example by using things other than the hf transformers library
# https://www.reddit.com/r/LocalLLaMA/comments/1djm0uo/is_there_any_reason_not_to_use_huggingface_and/
# https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/inference/model_utils.py

def load_model_tokenizer(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model     = AutoModelForCausalLM.from_pretrained(
                    **config.model_kwargs,
                    device_map   =  "auto")
    print(model.hf_device_map)
    #exit()
    
    tokenizer = AutoTokenizer.from_pretrained(
                    **config.model_kwargs,
                    padding_side = 'left')
    tokenizer.pad_token_id = model.config.eos_token_id[0]

    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    return model, tokenizer


def generate(config, model, tokenizer, prompts):

    input_ids = tokenizer(prompts, return_tensors="pt", truncation = True, padding = True).to("cuda")
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**input_ids, **config.generate_kwargs)
    
    return tokenizer.batch_decode(outputs, **config.tokenizer_kwargs)