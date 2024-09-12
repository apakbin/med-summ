import re
import utils
from types import SimpleNamespace

llm_to_prompt_format = {
    #https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
    "Llama_3.1": {
        "start":        "<|begin_of_text|>",
        "end":          "<|end_of_text|>",
        #"pad":         "<|finetune_right_pad_id|>",
        "header_start": "<|start_header_id|>",
        "header_end":   "<|end_header_id|>",
        #"mss_end":     "<|eom_id|>",
        "turn_end":     "<|eot_id|>",
        "py_tag":       "<|python_tag|>",

        ## role definitions
        "sys":          "system",
        "usr":          "user",
        "ipy":          "ipython",
        "llm":          "assistant"
    }
}


def get_model_version(config):
    """ Given a config file, returns model and version of the language model.
        Returns the output as:
        LLM_version (e.g. Llama_3.1)        
    """
    model_version = config.model_kwargs['model'].split('/')[-1]   # get the model and version by looking on the right of the last '/' charachter
    matcher       = re.compile(f"-.*-[+-]?([0-9]*[.])?[0-9]+-")   # match patterns of the form '-?-float-'
    is_matched    = matcher.search(model_version)
    assert is_matched, ("ERROR: Error parsing the model"          # if no match, throw an error
                       f" name and version from {config.pipeline_kwargs['model']}."
                        " Please make sure it has a structure similar to "
                        "'meta-llama/Meta-Llama-3.1-8B-Instruct'"
                        " where MODEL-VERSION(INT/FLOAT) occurs after the last '/' character.")
    
    return '_'.join([substr for substr in is_matched.group().split('-') if substr])


def assert_model_version(model_version, supported = llm_to_prompt_format.keys()):
    """ Checks whether the model and version is supported. This is mainly to make sure
    prompts are in the right format.
    """
    assert model_version in supported, f"ERROR: Seems like a model other than the supported ones in {supported} is used!"


def _combine_role_msg(prompt_format, role, msg = ''):
    """ 
    format: a dictionary containing role -> 
    Returns a string by combining role and message with proper headers.
    Output of the form: 
        > HEADER START + ROLE + HEADER END + MESSAGE
    """
    if msg is None:
        return ''
    return (
          prompt_format.header_start   # start of the header
        + getattr(prompt_format, role) # name of the role as recognized by the specific llm
        + prompt_format.header_end     # end of the header
        + msg )                        # actual message


def format_prompts(config, prompts):
    """ Formats a list of prompts for a language model, adding special tokens while
    ensuring the total token count stays within the model's supported limit. 
    Returns the output as:
        > [formatted prompt 1, formatted prompt 2, ...]
    """
    assert prompts, "ERROR: prompts cannot be empty!"

    model_version = get_model_version(config)             # get model version from config
    assert_model_version(model_version)                   # ensure model version supported for prompting

    tokenizer = utils.get_tokenizer(
        config.model_kwargs['model'])                     # get tokenizer based on model

    sys_msg = utils.read_file(config.sys_msg_file)        # read the system message from file

    prompt_format = SimpleNamespace(                      # get prompt formatter dictionary,
        **llm_to_prompt_format[model_version])            # convert to namespace for easier access

    preamble = (                                          # (get prompt preamble)
       _combine_role_msg (prompt_format, 'sys', sys_msg)) # system message
    
    epilogue = (                                          # (get prompt epilogue)
       _combine_role_msg (prompt_format, 'llm', ''))      # llm header, start of the agent response
    
    pre_epi_len = len(utils.tokenize(
        tokenizer, [preamble+epilogue])['input_ids'][0])  # get combined length of preamble and epilogue (computed in terms of tokens)
    
    max_user_msg_len = (                                  # maximum length the user message can be (computed in terms of tokens)
        config.model_kwargs['max_seq_len_to_capture']     # maximum sequence length accepted by model
      - pre_epi_len                                       # minus the combines length of preamble and epilogue
      - config.generate_kwargs['max_tokens'])             # minus the maximum number of tokens to be generated

    assert max_user_msg_len > 10, (                       # assert there remains enough space
        "ERROR: Not enough context length to include"     # to add at least a few tokens from
        " prompt. Consider increasing"                    # user input
        " max_seq_len_to_capture, reducing the sys"
        " message size, or decreasing the max_tokens"
        " in config.json.")
    
    formatted_prompts = [                                 # format prompts by
        _combine_role_msg (prompt_format, 'usr', prompt)  # adding user header to each
            for prompt in prompts]
    
    tokenized_prompts = utils.tokenize(                   # tokenize the formatted prompts
        tokenizer, formatted_prompts)['input_ids']        # we need them tokenized so we know how much
                                                          # to cut down to fit.
    
    cut_tokenized_prompts = [                             # ensure the tokenized versions are at most
        tknz_prmpt[:max_user_msg_len]                     # of length max_user_msg_len when tokenized
        for tknz_prmpt in tokenized_prompts]
    
    return [                                              # return the formatted prompts, each containing:
        ( preamble                                        # the preamble
       +  utils.detokenize(tokenizer, tknz_prmpt[1:])     # detokenized version of the tokenized prompt we cut to size. using [1:] to exclude 'start' special character. It gets added again later.
       +  epilogue)                                       # the epilogue
       for tknz_prmpt in cut_tokenized_prompts]