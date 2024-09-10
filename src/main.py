import utils
import infer
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
    measurement_dataset, note_dataset = notes.get_datasets('train')
    """ # PLOT TOKENIZED LENGTHS
    texts  = note_dataset.data["TEXT"].tolist()
    tokens = notes.tokenize(texts)["input_ids"]
    plot_n_token_hist(tokens, "./tmp/token_len_hist.jpg", xmax = 6000, bert_clen = 256)
    exit()
    """
    config  = utils.get_config()
    utils.set_visible_cuda_devices(config)
    prompts = prompt.format_prompts(
        config, 
        [
            {"usr": "Who is a good boy?"},
            {"sys": "Cutting Knowledge Date: December 2023;Today Date: 23 July 2024;You are a helpful assistant",
             "usr": "What is the capital of France? Give one word only, no sentences."},
        ])
    
    print (prompts)
    pipeline = infer.load_pipeline(config)
    
    print (infer.infer(config, pipeline, prompts))

if __name__=="__main__":
    main()