import utils
import infer
import generate
import prompt
import notes

def plot_n_token_hist(tokens, figaddr, **kwargs):
    import matplotlib.pyplot as plt
    #plt.figure(figsize=(19,12), dpi=300)
    plt.figure(dpi=300)
    plt.hist([len(tkn) for tkn in tokens], bins=50, color='k', edgecolor='black')

    if 'xmax' in kwargs:
        plt.xlim([0, kwargs['xmax']])

    if 'bert_clen' in kwargs:
        plt.axvline(
            x     = kwargs['bert_clen'], 
            color = 'r', linestyle = ':', linewidth=3, label = 'BERT Context Length')
        
    plt.xlabel('Length of Tokenized Notes')
    plt.ylabel('Frequency')        
        
    plt.legend()
    plt.savefig(figaddr)


#TODO: do we need to develop code to clean up the output? For example llama3.1 generates '\n\nParis'

def main():
    """
    infers = utils.load_pickle('./rslts/infers.pkl')
    for output in infers:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        print ("~~~ ~~~ ~~~ " * 5)
        print (" __PROMPT__ ")
        print (prompt)
        print (" __GENERATED__ ")
        print (generated_text)
        #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")#print (infers)
    exit()
    """
    
    """
    measurement_dataset, note_dataset = notes.get_datasets('train')
    """
    """ # PLOT TOKENIZED LENGTHS
    texts  = note_dataset.data["TEXT"].tolist()
    tokens = notes.tokenize(texts)["input_ids"]
    plot_n_token_hist(tokens, "./rslts/token_len_hist.jpg", xmax = 6000, bert_clen = 256)
    exit()
    """
    config  = utils.get_config()
    utils.set_visible_cuda_devices(config)
    """
    texts  = note_dataset.data["TEXT"].tolist()
    """
    texts  = utils.load_pickle('tmp/__text__.pkl')#[:5]
    #print (generate.generate(config, texts)[:5])
    #exit()

    prompts = prompt.format_prompts(
        config, 
        [
            {"sys": "You are a medical professional with expertise in summarizing ICU clinical notes accurately and concisely. Your task is to summarize the notes, including only the most critical information. If any part of the text is unclear or irrelevant, omit it from the summary. Limit your summary to a maximum of 230 tokens. Provide only the summary, without including phrases such as 'Here is a summary.'",
             "usr": texts[i]}
        for i in range(len(texts))])
    
    #model, tokenizer = infer.load_model_tokenizer(config)
    import time
    start = time.time()
    #infers = infer.generate(config, model, tokenizer, prompts)
    gens = generate.generate(config, prompts)
    duration = time.time() - start

    utils.dump_pickle(gens, './rslts/infers.pkl')

    print (f"""it took {duration} seconds to infer for {len(texts)} prompts.
           {round(duration/len(texts), 2)} seconds on average.
           """)

if __name__=="__main__":
    main()