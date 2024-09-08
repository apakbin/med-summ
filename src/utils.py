import os
import json
from types import SimpleNamespace

def get_config():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')) as f:
        return SimpleNamespace(**json.load(f))