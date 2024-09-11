from vllm import LLM, SamplingParams


def generate(config, prompts):
    model = LLM(**config.model_kwargs, tensor_parallel_size = len(config.gpus.split(',')))
    gens  = model.generate(prompts, 
                           sampling_params = SamplingParams(**config.generate_kwargs))
    return [gen.outputs[0].text for gen in gens]