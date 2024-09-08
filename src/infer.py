import torch
import transformers

#TODO: quantization, if needed, does not work with pipeline: 
# https://stackoverflow.com/questions/70609579/use-quantization-on-huggingface-transformers-models

#TODO: leaving the inference part simple for now, but might need to look into how to speed it up
# for example by using things other than the hf transformers library
# https://www.reddit.com/r/LocalLLaMA/comments/1djm0uo/is_there_any_reason_not_to_use_huggingface_and/
# https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/inference/model_utils.py

def load_pipeline(config):
    return transformers.pipeline(
        **config.pipeline_kwargs,
        model_kwargs = {"torch_dtype": torch.bfloat16}, 
        device_map   = "auto"
    )

def infer(config, pipeline, prompts):
    return pipeline(prompts, **config.generate_kwargs)