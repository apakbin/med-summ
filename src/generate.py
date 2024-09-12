import os
import math
import utils
import logging
import warnings
from vllm import LLM, SamplingParams
logging.getLogger("vllm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked shared_memory objects to clean up at shutdown")


def generate(config, prompts):
    model = LLM(**config.model_kwargs, tensor_parallel_size = len(config.gpus.split(',')))

    batch_fname = str(os.path.join(os.getcwd(), config.tmp_file, f"gen_batch_%d_BS_{config.gen_batch_size}.pkl"))
    n_batches   = math.ceil(len(prompts) / config.gen_batch_size)

    for i, batch in enumerate(utils.chunks(prompts, config.gen_batch_size)):
        batch_fname_ = batch_fname % i
        if os.path.isfile(batch_fname_):
            print (f"batch {i} out of {n_batches} already exists!")
            continue
        else:
            print (f"generating for batch {i} out of {n_batches}.")
            b_gens       = model.generate(
                                        batch, 
                                        sampling_params = SamplingParams(**config.generate_kwargs))
            b_gens       = [gen.outputs[0].text for gen in b_gens]
            utils.dump_pickle(b_gens, batch_fname_)
        
    # a somewhat cryptic way of flattening a list of lists: [[1, 2], [3], [4, 5]] -> [1, 2, 3, 4, 5]
    return [gen for gen_batch in [utils.load_pickle(batch_fname%i) for i in range(n_batches)] 
            for gen in gen_batch]