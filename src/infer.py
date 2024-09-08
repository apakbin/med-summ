import torch
import transformers

def load_pipeline(config):
    return transformers.pipeline(
        **config.pipeline_kwargs,
        model_kwargs = {"torch_dtype": torch.bfloat16}, 
        device_map   = "auto"
    )