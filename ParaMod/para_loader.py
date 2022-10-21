import json
import os
import logging
import pickle as pkl

import torch
from torch.utils.data.dataset import Dataset


paraphrase_root = '/home/ljx/Megatron-LM-main/tests/new_t5/output/new_bookcorpus/run_311'
cache_list = ['input_ids.pkl', 'attention_mask.pkl', 'labels.pkl']

class Para_dataset(Dataset):
    def __init__(self, tokenizer, file_list, processed_path, device='cuda', is_inference=False, finetune=False, sample_num=500):
        self.tokenizer = tokenizer
        self.file_list = file_list
        self.device = device
        self.max_length = 512
        self.processed_path = processed_path
        self.finetune = finetune
        self.sample_num = sample_num
        if self.finetune:
            self.processed_path = os.path.join(self.processed_path, 'finetune', str(self.sample_num))
        self.is_inference = is_inference
        
        if self.check_exist():
            self.load_cache()
        else:
            self.load_data()
            self.store_cache()
    
    def file_path(self):
        return [ os.path.join(self.processed_path, f) for f in cache_list ]
    
    def check_exist(self):
        return os.path.exists(self.file_path()[0]) and \
               os.path.exists(self.file_path()[1]) and \
               os.path.exists(self.file_path()[2])
    
    def load_cache(self):
        logging.info("Loading cache from %s" % self.processed_path)
        print("Loading cache from %s" % self.processed_path)
        cache = self.file_path()
        with open(cache[0], 'rb') as f:
            self.input_ids = pkl.load(f)
        with open(cache[1], 'rb') as f:
            self.attention_mask = pkl.load(f)
        with open(cache[2], 'rb') as f:
            self.labels = pkl.load(f)
    
    def store_cache(self):
        logging.info("Storing cache in %s" % self.processed_path)
        print("Storing cache in %s" % self.processed_path)
        cache = self.file_path()
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)
        with open(cache[0], 'wb') as f:
            pkl.dump(self.input_ids, f)
        with open(cache[1], 'wb') as f:
            pkl.dump(self.attention_mask, f)
        with open(cache[2], 'wb') as f:
            pkl.dump(self.labels, f)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_ids = self.input_ids[idx, :]#.to(self.device)
        samples = {
            'input_ids': input_ids,
        }
        
        print(samples['input_ids'].shape)
        if not self.is_inference:
            samples['attention_mask'] = self.attention_mask[idx, :]
            samples['labels'] = self.labels[idx, :]
            print(samples['attention_mask'].shape)
            print(samples['labels'].shape)
        exit()
        return samples
    
    def load_data(self):
        logging.info("Loading data from scratch from {}".format(self.processed_path))
        print("Loading data from scratch from {}".format(self.processed_path))
        all_data = []
        
        if self.finetune:
            for file in self.file_list:
                if file[0] == '.':
                    continue
                with open(file, 'r') as f:
                    count = 0
                    for line in f:
                        pair = line.strip().split("[JOIN]")
                        if len(pair) == 2:
                            all_data.append(line.split("[JOIN]"))
                            count += 1
                        if count>= self.sample_num:
                            break
        else:
            if len(self.file_list) == 0:
                self.file_list = os.listdir(paraphrase_root)
            for file in self.file_list:
                if file[0] == '.':
                    continue
                with open(os.path.join(paraphrase_root , file), 'r') as f:
                    data = json.load(f)
                data = data[1:]
                for entry in data:
                    all_data.append([entry['X'], entry['Y']])
            all_data = all_data[:self.sample_num]

        tokens_list, labels_list = [], []
        
        for pair in all_data:
            try:
                input_sentence, target = pair[0], pair[1]
            except IndexError:
                print(pair)
                continue
            tokens, labels = self.formatting(input_sentence, target)
            tokens_list.append(tokens)
            labels_list.append(labels)
        
        sentences = [self.tokenizer.decode(tokens)
                     for tokens in tokens_list]

        encodings = self.tokenizer(
            sentences, return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_length)
        
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        
        if not self.is_inference:
            self.labels = torch.tensor(labels_list, dtype=torch.long)
        
    def formatting(self, input_text, target_text):
        input_tokens = self.tokenizer.encode(input_text)
        target_tokens = self.tokenizer.encode(target_text)
        
        tokens = input_tokens 

        labels = target_tokens + [-100] * (self.max_length - len(target_tokens))
        labels = labels[:self.max_length]
        return tokens, labels
        
if __name__=='__main__':
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('/data/MODELS/t5-large')
    
    dataset = Para_dataset(tokenizer, file_list)
        