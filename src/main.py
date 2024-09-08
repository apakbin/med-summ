import transformers
import torch
import utils

import os
from huggingface_hub import login

login(token = os.environ['hf_token'])

config                             = utils.get_config()
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus)

pipeline = transformers.pipeline(
    "text-generation", model=config.model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

print (pipeline("Hey how are you doing today?", max_new_tokens = 20))