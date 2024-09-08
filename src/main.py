import utils
import infer
import prompt

def main():
    config = utils.get_config()
    utils.set_visible_cuda_devices(config)
    prompts = prompt.format_prompts(config, 
                          [
                              {   
                                  "usr_msg": "Who is a good boy?"
                               },
                              {   
                                  "sys_msg": "Cutting Knowledge Date: December 2023;Today Date: 23 July 2024;You are a helpful assistant",
                                  "usr_msg": "What is the capital of France? Give one word only, no sentences."
                               },
                               ])
    print (prompts)
    pipeline = infer.load_pipeline(config)
    
    print (infer.infer(config, pipeline, prompts))

if __name__=="__main__":
    main()