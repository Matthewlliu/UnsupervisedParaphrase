import json
import os
import pickle as pkl
from tqdm import tqdm

# process mscoco captions
coco_root = '/data/paraphrase/mscoco'
file = ['captions_train2017.json', 'captions_val2017.json']

#train_source_file = os.path.join(coco_root, file[0])
val_source_file = os.path.join(coco_root, file[1])

#with open(train_source_file, 'r') as f:
#    train_data = json.load(f)
with open(val_source_file, 'r') as f:
    val_data = json.load(f)
    
#train_data = train_data['annotations']
val_data = val_data['annotations']

#train_dict = {}
#for entry in train_data:
#    if entry['image_id'] not in train_dict:
#        train_dict[entry['image_id']] = [entry['caption']]
#    else:
#        train_dict[entry['image_id']].append(entry['caption'])
        
val_dict = {}
for entry in val_data:
    if entry['image_id'] not in val_dict:
        val_dict[entry['image_id']] = [entry['caption']]
    else:
        val_dict[entry['image_id']].append(entry['caption'])

output_path = '/data/ljx/data/paraphrase/mscoco/processed'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# random selection:
'''
import numpy as np
final_train = {}
for id, sentences in train_dict.items():
    index = np.random.choice(len(sentences), 2, replace=False)
    final_train[id] = [sentences[index[0]], sentences[index[1]]]
output_file = os.path.join(output_path, 'train_random.json')
with open(output_file, 'w') as f:
    json.dump(final_train, f)

final_val = {}
for id, sentences in val_dict.items():
    index = np.random.choice(len(sentences), 2, replace=False)
    final_val[id] = [sentences[index[0]], sentences[index[1]]]
output_file = os.path.join(output_path, 'val_random.json')
with open(output_file, 'w') as f:
    json.dump(final_val, f)
'''

'''
# semantic selection:
from bert_scorer import score
from bert_score.utils import get_model as get_sem_model
from bert_score.utils import get_tokenizer as get_sem_tokenizer
from bert_score.utils import model2layers

sem_model_path = '/data/ljx/cpt/roberta-large-mnli'
sem_model_name = sem_model_path.split('/')[-1]
sem_model_name = sem_model_name.split('_')[0]
num_layers = model2layers[sem_model_name]
sem_model = get_sem_model(sem_model_path, num_layers, False)
sem_tokenizer = get_sem_tokenizer(sem_model_path, False)

device = 'cuda'
sem_model = sem_model.cuda()

def score_sentences(sents):
    length = len(sents)
    f1_max = -1
    to_return = [0,1]
    for i in range(length):
        for j in range(i+1, length):
            _, _, f1 = score( [sents[i]], [sents[j]], model=sem_model, tokenizer=sem_tokenizer, lang='en')
            if f1 > f1_max:
                f1_max = f1
                to_return = [i, j]
    return to_return
'''

'''
final_train = {}
for id, sentences in tqdm(train_dict.items()):
    index = score_sentences(sentences)
    final_train[id] = [sentences[index[0]], sentences[index[1]]]
output_file = os.path.join(output_path, 'train_bertscore.json')
with open(output_file, 'w') as f:
    json.dump(final_train, f)
'''

final_val = {}
for id, sentences in tqdm(val_dict.items()):
    #index = score_sentences(sentences)
    final_val[id] = [sentences[0], sentences[1:]]
output_file = os.path.join(output_path, 'val_all.json')
with open(output_file, 'w') as f:
    json.dump(final_val, f)