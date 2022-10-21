from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import os
import numpy as np
device = 'cuda'

def ppl_score(sentences, eva_model, eva_tokenizer):
    max_length = eva_model.config.n_positions
    stride = 1

    sentences_id = [ eva_tokenizer(sen, return_tensors='pt').input_ids for sen in sentences ]
    
    ppls = []
    for sentence in sentences_id:
        nlls = []
        if sentence.size(1) == 1:
            ppl = torch.tensor(-10)
        else:
            for i in range(0, sentence.size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, sentence.size(1))
                trg_len = end_loc - i    # may be different from stride on last loop
                input_ids = sentence[:,begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:,:-trg_len] = -100

                with torch.no_grad():
                    outputs = eva_model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs[0] * trg_len

                nlls.append(neg_log_likelihood)

            nlls = nlls[1:]
            ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        ppls.append(ppl.item())
    ppls = 1/np.array(ppls)
    ppls = 1/(1+np.exp(-4*ppls)) - 0.5
    return ppls

if __name__=='__main__':
    sent = ['my heart was racing so fast that it might explode right out of my chest .', 
        'I one hub, sinners in me and crabs house more loving lone longers. ']
    
    device = 'cuda'
    model_id = 'gpt2-large'
    model = GPT2LMHeadModel.from_pretrained(os.path.join('/data/MODELS', model_id)).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    
    ppl = ppl_score(sent, model, tokenizer)
    print(ppl)