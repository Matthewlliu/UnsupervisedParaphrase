import math
import os
import logging

import torch
#from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


class gpt2_model(object):
    def __init__(self, args):
        self.args = args
        self.special_tokens_dict = {'sep_token': '[SEP]'}
        self.device = self.args.device
        self.model = self.tokenizer = None
        self.global_step = None
        self.c_tokenizer = TreebankWordTokenizer()
        self.c_detokenizer = TreebankWordDetokenizer()
        self.english_stopwords = stopwords.words('english')

    def build_model(self, checkpoint_dir=None, with_tokenizer=False):
        if checkpoint_dir is None or with_tokenizer is False:
            self.tokenizer = GPT2Tokenizer.from_pretrained('/data/MODELS/%s' % self.args.model)
            if self.special_tokens_dict is not None:
                self.tokenizer.add_special_tokens(self.special_tokens_dict)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Load {} tokenizer".format(self.args.model))
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
            logging.info("Load tokenizer from {}".format(checkpoint_dir))

        if checkpoint_dir is None:
            self.model = GPT2LMHeadModel.from_pretrained('/data/MODELS/%s' % self.args.model)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info("Load {} model".format(self.args.model))
        else:
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
            logging.info("Load model from {}".format(checkpoint_dir))
        self.model.to(self.device)
        self.model.train()

        if hasattr(self.args, 'summary_dir'):
            self.writer = SummaryWriter(self.args.summary_dir)

    def remove_stopwords(self, sentence):
        sentence = self.c_tokenizer.tokenize(sentence)
        sentence = [word for word in sentence
                    if word.lower() not in self.english_stopwords]
        sentence = ' '.join(sentence)
        sentence = sentence.replace("''", '"').replace('``', '"')
        sentence = self.c_detokenizer.detokenize(sentence.split())
        return sentence
    
    def filter_special_tokens(self, sent, eos='<|endoftext|>'):
        while sent.endswith(eos):
            sent = sent[:-len(eos)].strip()
        return sent
            
    def generate_text(self, input_texts, max_length=512, decoding='greedy', suffix='', isfilter=False):
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
            for input_text in input_texts: #tqdm(input_texts):
                sequences = []
                input_text = input_text.strip()
                input_text += suffix
                input_text = self.remove_stopwords(input_text)
                #logging.info('Start to generate from "{}"'.format(input_text))
                input_encoding = self.tokenizer.encode(
                    input_text, return_tensors='pt')
                input_encoding = input_encoding.to(self.device)
                generated_tokens = self.model.generate(
                    input_encoding, **kwargs)
                for tok_seq in generated_tokens:
                    sequence = self.tokenizer.decode(tok_seq)
                    #logging.info("Generated text: {}".format(sequence))
                    if isfilter is True:
                        sequence = self.filter_special_tokens(sequence)
                    sequence = sequence.split(suffix)[1]
                    sequences.append(sequence)
                sentences_list.append(sequences)
        return sentences_list