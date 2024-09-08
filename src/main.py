import utils
import infer

def main():
    config = utils.get_config()
    utils.set_visible_cuda_devices(config)
    pipeline = infer.load_pipeline(config)

    print (config.generate_kwargs)
    print (pipeline(["Hey how are you doing today?", 
                     """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Cutting Knowledge Date: December 2023;Today Date: 23 July 2024;You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>What is the capital of France? Give one word only, no sentences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""], **config.generate_kwargs))

if __name__=="__main__":
    main()
    #https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/inference/model_utils.py