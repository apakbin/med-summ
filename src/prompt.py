import re
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
    """ Given a config file, returns model and version of the language model."""
    model_version = config.pipeline_kwargs['model'].split('/')[-1]
    matcher       = re.compile(f"-.*-[+-]?([0-9]*[.])?[0-9]+-")
    is_matched    = matcher.search(model_version)
    assert is_matched, f"""ERROR: Error parsing the model name and version from {config.pipeline_kwargs['model']}.
    Please make sure it has a structure similar to 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    where MODEL-VERSION occurs after the last '/' character."""
    return '_'.join([substr for substr in is_matched.group().split('-') if substr])


def assert_model_version(model_version, supported = ["Llama_3.1"]):
    """ Checks whether the model and version is supported. This is mainly to make sure
    prompts are in the right format.
    """
    assert model_version in supported, f"ERROR: Seems like a model other than the supported ones in {supported} is used!"


def format_prompt(model_version, prompt):
    """Expects a model_version, and a prompt which is a dictionary of the form:
    {'sys_msg':, 'usr_msg':}. 
    The output will look like: (extra formatting omitted)
        > SYS MESSAGE
        > USER MESSAGE
    
    If (any of) keys is not present, the corresponding part is omitted from the output.
    """
    format = SimpleNamespace(**llm_to_prompt_format[model_version])

    def _encapsulate(role, msg = ''):
        """ Returns a string by combining role and message with proper headers.
        Output of the form: 
            > HEADER START + ROLE + HEADER START + MESSAGE
        """
        if msg is None:
            return ''
        return format.header_start + getattr(format, role) + format.header_end + msg
    
    return (
       format.start   # start of prompt
     + _encapsulate('sys', prompt.get('sys_msg', None)) # system message
     + _encapsulate('usr', prompt.get('usr_msg', None)) # user message
     + _encapsulate('llm', '') # add llm header
    )


def format_prompts(config, prompts):
    """ Formats a list of prompts for a language model.
    """
    model_version = get_model_version(config)
    assert_model_version(model_version)
    return [format_prompt(model_version, prompt) for prompt in prompts]