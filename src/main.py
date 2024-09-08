import transformers
import torch
import utils

config = utils.get_config()
utils.set_visible_cuda_devices(config)


pipeline = transformers.pipeline(
    "text-generation", model=config.model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

print (pipeline("Hey how are you doing today?", max_new_tokens = 20))