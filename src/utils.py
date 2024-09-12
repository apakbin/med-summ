import os
import json
import pickle
import functools
from pathlib import Path
from types import SimpleNamespace
from huggingface_hub import login as _hf_login


def hf_login():
    _hf_login(token = os.environ['hf_token'])


def set_visible_cuda_devices(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus)


def get_config():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')) as f:
        return SimpleNamespace(**json.load(f))


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dump_pickle(obj, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


#https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def cache(func):
    @functools.wraps(func)

    def _cache(file_name, func, *args, **kwargs):

        CACHE_ADDRESS = os.path.join(os.getcwd(), 'cache/')
        file_name_ = file_name+'.pkl'
        file_path = os.path.join(CACHE_ADDRESS, file_name_)

        if Path(file_path).is_file():
            return load_pickle(file_path)

        else:
            create_dir_if_not_exist(os.path.dirname(file_path))
            output = func(*args, **kwargs)

            dump_pickle(output, file_path)

            return output

    def _func_args_to_str(func, *args, **kwargs):
        output = func.__name__
        for arg in args:
            output += "__"+str(arg)

        for key, val in kwargs.items():
            output += "__"+str(key)+"_"+str(val)

        return output

    def cache(*args, **kwargs):
        file_name = _func_args_to_str(func, *args, **kwargs)
        return _cache(file_name, func, *args, **kwargs)

    return cache