import os
import json
from huggingface_hub import login as _hf_login
from types import SimpleNamespace


def hf_login():
    _hf_login(token = os.environ['hf_token'])

def set_visible_cuda_devices(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus)

def get_config():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')) as f:
        return SimpleNamespace(**json.load(f))