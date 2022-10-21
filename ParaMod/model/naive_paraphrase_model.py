import math
import os
import logging

import torch
#from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace

class naive_model(object):
    def __init__(self, args):
        self.args = args
        self.special_tokens_dict = None #{'sep_token': '[SEP]'}
        self.device = self.args.device
        self.model = self.tokenizer = None
        self.global_step = None
        
    def build_model(self, checkpoint_dir=None, with_tokenizer=False):
        if checkpoint_dir is None or with_tokenizer is False:
            self.tokenizer = BertTokenizer.from_pretrained('/data/MODELS/%s' % self.args.model)
            if self.special_tokens_dict is not None:
                self.tokenizer.add_special_tokens(self.special_tokens_dict)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Load {} tokenizer".format(self.args.model))
        else:
            self.tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
            logging.info("Load tokenizer from {}".format(checkpoint_dir))

            
        if checkpoint_dir is None:
            self.model = BertForMaskedLM.from_pretrained('/data/MODELS/%s' % self.args.model)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info("Load {} model".format(self.args.model))
        else:
            self.model = BertForMaskedLM.from_pretrained(checkpoint_dir)
            logging.info("Load model from {}".format(checkpoint_dir))
            
        #if torch.cuda.device_count()>1:
        #    self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.train()
        
        if hasattr(self.args, 'summary_dir'):
            self.writer = SummaryWriter(self.args.summary_dir)

    def Paraphrase(self, input_texts):
        insert_prob = 0.15
        delete_prob = 0.1
        replace_prob = 0.30
        
        def insert(tokens, pos):
            input_text = ' '.join(tokens[:pos]) + ' [MASK] ' + ' '.join(tokens[pos:])
            inputs = self.tokenizer(input_text.strip(), return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            outs = self.tokenizer.decode(predicted_token_id)
            #output = input_text.replace('[MASK]', outs)
            #tokens = output.strip().split()
            tokens = tokens[:pos] + [outs] + tokens[pos:]
            return tokens
        
        def delete(tokens, pos):
            try:
                tokens[pos] = ''
            except IndexError:
                print(tokens)
                print(pos)
            return tokens
        
        def replace(tokens, pos):
            input_text = ' '.join(tokens[:pos]) + ' [MASK] ' + ' '.join(tokens[(pos+1):])
            inputs = self.tokenizer(input_text.strip(), return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            outs = self.tokenizer.decode(predicted_token_id)
            #output = input_text.replace('[MASK]', outs)
            #tokens = output.strip().split()
            tokens[pos] = outs
            return tokens
        
        self.model.eval()
        sentences_list = []
        for input_text in input_texts:
            input_text = input_text.replace('?', ' ?')
            input_text = input_text.replace('.', ' .')
            input_text = input_text.replace('!', ' !')
            input_text = input_text.replace(',', ' ,')
            sequences = []
            for num in range(self.args.augment_size-1):
                tokens = input_text.strip().split()
                # begin
                pos = 0
                while(pos < len(tokens)):
                    rnd1 = np.random.uniform(0,1)
                    rnd2 = np.random.uniform(0,1)
                    if rnd1 <= insert_prob:
                        tokens = insert(tokens, pos)
                        pos += 1
                    if rnd2 <= delete_prob:
                        tokens = delete(tokens, pos)
                    elif rnd2 > delete_prob and rnd2 <=(delete_prob+replace_prob):
                        tokens = replace(tokens, pos)
                        
                    pos += 1

                rnd = np.random.uniform(0,1)
                if rnd <= insert_prob:
                    tokens = insert(tokens, len(tokens)) # insert at the end
                sequence = ' '.join(tokens)
                sequences.append(sequence)
            sentences_list.append(sequences)
        return sentences_list
    
if __name__=='__main__':
    args  = {
        'model': 'bert-base-uncased',
        'device': 'cuda',
        'augment_size': 3,
        'ckpt': '/data/MODELS/bert-base-uncased'
    }
    args = SimpleNamespace(**args)
    lm = naive_model(args)
    lm.build_model(checkpoint_dir=args.ckpt)
    
    inputs = ['I am Paul McCartney from the famous rock band the Beatles.', 
             'Well nice to meet you, my name is John Lennon.']
    print(lm.Paraphrase(inputs))