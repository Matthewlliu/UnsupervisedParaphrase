import torch
import transformers
import copy

import torch.nn as nn
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch.distributed as dist

class MyModel(object):
    def __init__(self, model_name, pretrained_dir=None):
        super().__init__()

        # The tokenizer. Megatron was trained with standard tokenizer(s).
        if model_name[:4] == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            if pretrained_dir is not None:
                print("Loading model from %s" % pretrained_dir)
                self.model = GPT2LMHeadModel.from_pretrained(pretrained_dir)
            else:
                self.model = GPT2LMHeadModel.from_pretrained(model_name)
        elif model_name[:4] == 'bart':
            self.tokenizer = BartTokenizer.from_pretrained('facebook/%s' % model_name)
            if pretrained_dir is not None:
                print("Loading model from %s" % pretrained_dir)
                self.model = BartForConditionalGeneration.from_pretrained(pretrained_dir)
            else:
                self.model = BartForConditionalGeneration.from_pretrained('facebook/%s' % model_name)
        elif model_name[:2] == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            if pretrained_dir is not None:
                print("Loading model from %s" % pretrained_dir)
                self.tokenizer = T5Tokenizer.from_pretrained(pretrained_dir)
                self.model = T5ForConditionalGeneration.from_pretrained(pretrained_dir)
            else:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self._cuda_device = 'cpu'
        self.try_cuda()
        self.model.eval()

    def try_cuda(self):
        if torch.cuda.is_available():
            if self._cuda_device != 'cuda':
                self._cuda()
                self._cuda_device = torch.device("cuda")
        else:
            raise ValueError("Cuda Device Not Found!")

    def _cuda(self):
        #self.model.to(self._cuda_device)
        self.model.cuda()
    
    def encode(self, inp):
        return self.tokenizer(inp, return_tensors='pt').to(self._cuda_device) #for inp in sentences ]
