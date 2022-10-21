from datasets import load_dataset
import json
import os
from tqdm import tqdm

def get_new_book_names():
    data_path = '/data/ljx/data/book_corpus/'
    original_urls = os.path.join(data_path, 'url_list.jsonl.txt')
    new_urls = os.path.join(data_path, 'books1/2020-08-27-epub_urls.txt')

    original_content = []
    with open(original_urls, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)['epub']
            name = data.split('/')[-1]
            name = name.split('.')[0]
            original_content.append(name)

    new_content = []
    with open(new_urls, 'r') as f:
        data = f.readlines()

    for entry in data:
        name = entry.strip().split('/')[-1]
        name = name.split('.')[0]
        new_content.append(name)

    diff = [ entry for entry in new_content if entry not in original_content ]
    diff = list(set(diff))
    return diff

def get_new_book_wholenames():
    data_path = '/data/ljx/data/book_corpus/'
    original_urls = os.path.join(data_path, 'url_list.jsonl.txt')
    new_urls = os.path.join(data_path, 'books1/2020-08-27-epub_urls.txt')

    original_content = []
    with open(original_urls, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)['epub']
            name = data.split('/')[-1]
            #name = name.split('.')[0]
            original_content.append(name)

    new_content = {}
    with open(new_urls, 'r') as f:
        data = f.readlines()

    for entry in data:
        #entry = entry.strip()
        #if len(entry.strip().split('/')) != 11:
        #    print(entry)
        name = entry.strip().split('/')[-1]
        #name = name.split('.')[0]
        new_content[name] = entry.strip()
        #new_content.append(name)

    #diff = [ entry for entry in new_content if entry not in original_content ]
    diff = []
    for k,v in new_content.items():
        if k not in original_content:
            diff.append(v)
    diff = list(set(diff))
    return diff


if __name__=='__main__':
    diff = get_new_book_names()
    print(len(diff))
    