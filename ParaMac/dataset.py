import json
import pickle as pkl
import os
import copy

import numpy as np

file_name = 'bidirect_context_window{}_sample{}.pkl'
source_path = '/data/ljx/data/bookcorpus'
source_name = 'book_corpus_forward.jsonl'
total_length = 74004228
sample_posibility = 0.1

class para_data_bidirect():
    def __init__(self, data_path, context_length=3, sample_num=10000):
        self.data_path = data_path
        self.context_length = context_length
        self.sample_num = sample_num
        if self.check_exist():
            self.data = self.load_cache()
        else:
            self.data = self.sample_from_scratch()
            self.store_cache()
    
    def file_path(self):
        return os.path.join(self.data_path, file_name.format(self.context_length, self.sample_num))
    
    def check_exist(self):
        return os.path.exists(self.file_path())

    def sample_from_scratch(self):
        samples = []
        input_path = os.path.join(source_path, source_name)
        
        bank = [ '' for _ in range(2*self.context_length + 1)]
        
        with open(input_path, 'r') as f:
            count = 0
            for line in f:
                line = json.loads(line)
                bank.append(line['text'])
                bank = bank[1:]
                count += 1
                if count > len(bank):
                    break
            for line in f:
                line = json.loads(line)
                bank.append(line['text'])
                bank = bank[1:]
                if (len(bank[self.context_length]) >= 60) and (np.random.rand() < sample_posibility):
                    samples.append(copy.deepcopy(bank))
                if len(samples) >= self.sample_num:
                    break
                
        return samples
    
    def load_cache(self):
        print('Loading cache from %s' % self.file_path())
        with open(self.file_path(), 'rb') as f:
            data = pkl.load(f)
        return data
    
    def store_cache(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        with open(self.file_path(), 'wb') as f:
            pkl.dump(self.data, f)
        print('Caching into %s' % self.file_path())

        
def test():
    para = para_data_bidirect('/data/ljx/data/book_corpus/para', 3)
    print(len(para.data))
    print(para.data[0])
    #para = para_data('/data/ljx/data/book_corpus/para', 4)

if __name__=='__main__':
    test()