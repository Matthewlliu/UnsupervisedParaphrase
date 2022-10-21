import math
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

class t5_model(object):
    def __init__(self, args):
        self.args = args
        self.special_tokens_dict = None #{'sep_token': '[SEP]'}
        self.device = self.args.device
        self.model = self.tokenizer = None
        self.global_step = None
        
    def build_model(self, checkpoint_dir=None, with_tokenizer=False):
        if checkpoint_dir is None or with_tokenizer is False:
            self.tokenizer = T5Tokenizer.from_pretrained('/data/MODELS/%s' % self.args.model)
            if self.special_tokens_dict is not None:
                self.tokenizer.add_special_tokens(self.special_tokens_dict)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Load {} tokenizer".format(self.args.model))
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
            logging.info("Load tokenizer from {}".format(checkpoint_dir))

            
        if checkpoint_dir is None:
            self.model = T5ForConditionalGeneration.from_pretrained('/data/MODELS/%s' % self.args.model)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info("Load {} model".format(self.args.model))
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
            logging.info("Load model from {}".format(checkpoint_dir))
            
        #if torch.cuda.device_count()>1:
        #    self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.train()
        
        if hasattr(self.args, 'summary_dir'):
            self.writer = SummaryWriter(self.args.summary_dir)

    def generate_text(self, input_texts, max_length=512, decoding='sampling', suffix='', isfilter=False):
        self.model.eval()
        sentences_list = []
        with torch.no_grad():
            kwargs = {'max_length': max_length}
            if decoding == 'sampling':
                kwargs['do_sample'] = True
                kwargs['top_k'] = self.args.k
                kwargs['top_p'] = self.args.p
                kwargs['temperature'] = self.args.temperature
                kwargs['num_return_sequences'] = self.args.num_generate
                kwargs['pad_token_id'] = self.tokenizer.eos_token_id
            elif decoding == 'beam_gen':
                kwargs['do_sample'] = False
                kwargs['num_beams'] = self.args.beam_size
                kwargs['temperature'] = self.args.temperature
                kwargs['num_return_sequences'] = self.args.num_generate
                kwargs['pad_token_id'] = self.tokenizer.eos_token_id
            elif decoding == 'beam_sample':
                kwargs['do_sample'] = True
                kwargs['top_k'] = self.args.k
                kwargs['top_p'] = self.args.p
                kwargs['num_beams'] = self.args.beam_size
                kwargs['temperature'] = self.args.temperature
                kwargs['num_return_sequences'] = self.args.num_generate
                kwargs['pad_token_id'] = self.tokenizer.eos_token_id
            for input_text in tqdm(input_texts):
                sequences = []
                input_text = input_text.strip()
                #print("Input:", input_text)
                #print("Output:")
                input_text += suffix
                logging.info('Start to generate from "{}"'.format(input_text))
                input_encoding = self.tokenizer.encode(
                    input_text, return_tensors='pt')
                input_encoding = input_encoding.to(self.device)
                generated_tokens = self.model.generate(
                    input_encoding, **kwargs)
                for tok_seq in generated_tokens:
                    sequence = self.tokenizer.decode(tok_seq)
                    logging.info("Generated text: {}".format(sequence))
                    if isfilter is True:
                        sequence = self.filter_special_tokens(sequence)
                    #print(sequence)
                    sequences.append(sequence)
                    
                sentences_list.append(sequences)
        return sentences_list
    
    def filter_special_tokens(self, sent):
        sent = sent.strip().lower()
        if '<extra_id_0>' in sent:
            sent = sent.split('<extra_id_0>')[1]
        if '<extra_id_1>' in sent:
            sent = sent.split('<extra_id_1>')[0]
        while sent.endswith('</s>'):
            sent = sent[:-len('</s>')].strip()
        if sent[0] in ['"', '.']:
            sent = sent[1:]
        if sent[-2:] == '?.':
            sent = sent[:-1]
        sent = sent.replace('.', ' .')
        sent = sent.replace('?', ' ?')
        sent = ' '.join(sent.replace('<unk>', '').split(' '))
        return sent