import json
import pickle as pkl
import json
import os
import re
import copy
from nltk.tokenize import sent_tokenize

import numpy as np

from book_corpus_filter import get_new_book_names

file_name = 'newbook_context{}_sample{}_0705.pkl'
#file_name = 'test_context{}_sample{}.json'
source_path = '/data/ljx/data/book_corpus/books1/epubtxt'
#source_name = 'book_corpus_new.jsonl'

class bookcorpus_bidirect():
    def __init__(self, data_path, remake=False, context=200, sample_num=10000):
        self.data_path = data_path
        self.sample_num = sample_num
        self.context = context
        if self.check_exist() and not remake:
            self.data = self.load_cache()
        else:
            self.data = self.sample_from_scratch()
            self.store_cache()
    
    def file_path(self):
        return os.path.join(self.data_path, file_name.format(self.context, self.sample_num))
    
    def check_exist(self):
        return os.path.exists(self.file_path())

    def sample_from_scratch(self):
        def sentence_strip(sent):
            count = 0
            for c in sent:
                if c == '"':
                    count += 1
            if count%2 == 1:
                if sent[0] == '"':
                    sent = sent[1:]
                elif sent[-1] == '"':
                    sent = sent[:-1]
            return sent
        
        samples = []
        new_names = get_new_book_names()
        print(len(new_names))
        for book_names in new_names:
            book_path = os.path.join(source_path, book_names+'.epub.txt')
            if not os.path.exists(book_path):
                continue
            with open(book_path, 'r') as f:
                content = f.readlines()
            content = [ re.sub(u"\\(.*?\\)", "", e.strip()) for e in content if len(e.strip())>0 ]
            # divide by the length of paragraph
            label = []
            for sent in content:
                if len(sent)<60:
                    label.append(0)
                elif len(sent)<100:
                    label.append(1)
                else:
                    label.append(2)
                    
            # start sampling
            for i, sent in enumerate(content):
                if label[i]==0:
                    continue
                elif label[i]==1:
                    output = self.find_context(i, content, label)
                    if output is not None:
                        sent = sentence_strip(sent)
                        samples.append([output[0], sent.strip(), output[1]])
                else:
                    #special_dots = {'MR.':'<MR>', 'Mr.':'<Mr>', 'mr.':'<mr>',
                    #                'DR.', 'Dr.', 'dr.'}
                    #for sd in special_dots:
                    #    sent = sent.replace()
                    parts = sent_tokenize(sent)
                    #tmp = []
                    #for s in parts:
                    #    s = s + '.'
                    #    d = s.split('?')
                    #    if len(d) == 1:
                    #        tmp.extend(d)
                    #    else:
                    #        for dd in d:
                    #            tmp.append(dd+'?')
                    #        tmp[-1] = tmp[-1][:-1] # remove the last ?
                    #parts = 
                    for j, s in enumerate(parts):
                        if len(s) > 60 and len(s) < 100:
                            output = self.find_context(i, content, label, j, parts)
                            if output is not None:
                                sent = sentence_strip(s)
                                samples.append([output[0], sent.strip(), output[1]])
            
            if len(samples) >= self.sample_num:
                break
                
        return samples
    
    def find_context(self, i, content, label, j=None, parts=None):
        if j is None:
            pre = ''
            post = ''
        else:
            pre = ''.join(parts[:j])
            post = ''.join(parts[j+1:])
        # pre
        pre_flag = False
        count = i - 1
        while(count >= 0):
            if label[count]==0:
                return None
            else:
                pre = content[count] + ' ' + pre
            count -= 1
            # if to stop
            if len(pre) > self.context:
                pre_flag = True
                if len(pre) > 1.5*self.context:
                    pre = pre[-round(1.2*self.context):]
                break
        # post
        post_flag = False
        count = i + 1
        while(count<len(content)):
            if label[count]==0:
                return None
            else:
                post = post + ' ' + content[count]
            count += 1
            if len(post) > self.context:
                post_flag = True
                if len(post) > 1.5*self.context:
                    post = post[:round(1.2*self.context)]
                break
        if pre_flag is True and post_flag is True:
            return [pre.strip(), post.strip()]
        else:
            return None
        
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
        #with open(self.file_path(), 'w') as f:
        #    for entry in self.data:
        #        json.dump(entry, f)
        #        f.write('\n')
        print('Caching into %s' % self.file_path())

        
def test():
    from termcolor import colored
    para = bookcorpus_bidirect('/data/ljx/data/book_corpus/para', True, 250, 1000000)
    for i in range(10):
        print(para.data[i][0])
        print(colored(para.data[i][1], 'red'))
        print(para.data[i][2])
        print('\n')
        input()
    print(len(para.data))
    #para = para_data('/data/ljx/data/book_corpus/para', 4)

if __name__=='__main__':
    test()