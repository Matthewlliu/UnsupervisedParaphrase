import os
import torch
import numpy as np
from termcolor import colored
import pickle as pkl
import json
import copy
import math
import argparse

#from my_model import MyModel
from utils import kw_extraction, kw_sort, kw_filter,kw_substi,kw_substi_bert, Scoring, end_process, evaluation, get_input, get_output, filter_cands
#from dataset import para_data_bidirect
from bookcorpus_dataset import bookcorpus_bidirect
from model import MyModel

from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import BertTokenizer, BertForMaskedLM
from bert_score.utils import get_model as get_sem_model
from bert_score.utils import get_tokenizer as get_sem_tokenizer
from bert_score.utils import model2layers

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--step', type=int, default=125)
parser.add_argument('--test_id', type=str, required=True)
parser.add_argument('--sem_weight', type=float, default=4.0)
parser.add_argument('--ppl_weight', type=float, default=8.0)
parser.add_argument('--div_weight', type=float, default=1.2)
parser.add_argument('--kw_drop_per', type=float, default=0.3)
parser.add_argument('--kw_sub_per', type=float, default=1.0)
parser.add_argument('--kw_sub_temp', type=float, default=1.5)
parser.add_argument('--temperature', type=float, default=1.5)
parser.add_argument('--if_save', action='store_true')
parser.add_argument('--sem_model', type=str, default='/data/ljx/cpt/roberta-large-mnli', help='the huggingface model name')
args = parser.parse_args()

#max_length = 75
top_k = 50
top_p = 0.92
n_beams = 10
early_stopping = False
do_sample = False
temperature = args.temperature

kw_sub_per = args.kw_sub_per
kw_drop_per = args.kw_drop_per
kw_sub_temp = args.kw_sub_temp

permutation_try_max = 15
test_number = 10
if_save = args.if_save
score_k = 5

'''
data_path = '/data/ljx/data/book_corpus/para/'
Data = bookcorpus_bidirect(data_path, False, 250, 100000).data
check = 'with a swift flick red drops fell rapidly onto the tiled floor.'
for i, sample in enumerate(Data):
    if sample[1].lower() == check:
        print("Index: ", i)
exit()
'''

#model_name = 'bart-base'
model_name = 't5-large'
if_bart = model_name[:4]=='bart'

# loading all the models
print(colored("Loading generation model...", "red"))
#my_pretrained_root = '/data/ljx/cpt/KQAPro_ckpt/program_ckpt/'
my_pretrained_root = '/data/MODELS/%s' % model_name
my_model = MyModel(model_name, pretrained_dir = my_pretrained_root)

print(colored("Loading fluency evaluation model...", "red"))
model_id = 'gpt2'
gpt_model = GPT2LMHeadModel.from_pretrained(os.path.join('/data/MODELS', model_id)).cuda()
gpt_tokenizer = GPT2TokenizerFast.from_pretrained(os.path.join('/data/MODELS', model_id))

# substitution model
model_id = 'bert-base-uncased'
sub_model = BertForMaskedLM.from_pretrained(os.path.join('/data/MODELS', model_id)).cuda()
sub_tokenizer = BertTokenizer.from_pretrained(os.path.join('/data/MODELS', model_id))

print(colored("Loading semantic evaluation model...", "red"))
#t5_eval = './finetuning-t5/mnli/cache/t5-base_best'
#sem_model = args.sem_model
sem_model_name = args.sem_model.split('/')[-1]
sem_model_name = sem_model_name.split('_')[0]
num_layers = model2layers[sem_model_name]
sem_model = get_sem_model(args.sem_model, num_layers, False)
sem_tokenizer = get_sem_tokenizer(args.sem_model, False)

data_path = '/data/ljx/data/book_corpus/para/'
Data = bookcorpus_bidirect(data_path, False, 250, 1000000).data
data_use = Data[args.start:args.start+args.step]
#Data = para_data_bidirect(data_path)
#data_path = './ao3_test2.json'
#with open(data_path, 'r') as f:
#    data_use = json.load(f)

# output file:
output_path = './output/new_bookcorpus/run_0804'# % args.test_id #'/data/ljx/results/para/test_output_1_4'
if not os.path.exists(output_path):
    os.makedirs(output_path)
test_id = args.test_id
save_id = int(args.start/1000)

print(colored("Output (top-{}):".format(score_k), 'red'))
# Generate
print(colored("model       = {}".format(model_name), 'yellow'))
print(colored("top_k       = {}".format(top_k), 'yellow'))
print(colored("top_p       = {}".format(top_p), 'yellow'))
print(colored("n_beams     = {}".format(n_beams), 'yellow'))
print(colored("do_sample   = {}".format(do_sample), 'yellow'))
print(colored("temperature = {}".format(temperature), 'yellow'))
print("Start from {} to {}".format(args.start, args.start+args.step))

actual_s = []
generated_s = []

#data_use = Data.data[:test_number]
#data_use = data_use[1:]
json_dump = [vars(args)]
for idx, sample in tqdm(enumerate(data_use)):
    sample = [ v.lower() for v in sample ]
    
    context_len = len(sample)
    mid_id = round((context_len - 1)/2)
    X = sample[mid_id]

    #Y_ref = sample[mid_id-1:mid_id+1]
    actual_s.append(X)
    
    print(colored("Input: "+X, 'yellow'))
    kw_batches = kw_extraction(X)
    #kw_batches = kw_sort(X, kw_batches)
    kw_batches_ori = kw_filter(X, kw_batches, kw_drop_per, my_model)
    
    kw_num = len(kw_batches_ori)
    permutation_try = permutation_try_max if kw_num>3 else math.factorial(kw_num)-1
    if permutation_try == 0:
        permutation_try += 1
    permutation_cache = list() 
    print(colored("Key Words: "+', '.join(kw_batches_ori), 'yellow'))
    #kw_batches_tmp = kw_substi(sample, mid_id, kw_batches_ori, kw_sub_per, kw_sub_temp, my_model)
    kw_batches_tmp = kw_substi_bert(sample, mid_id, kw_batches_ori, kw_sub_per, kw_sub_temp, sub_model, sub_tokenizer)
    print(colored("Substituted: "+', '.join(kw_batches_tmp), 'yellow'))
    
    X_cands = []
    for i in range(permutation_try):
        inputs, X_processed, permutation_cache = get_input(sample, mid_id, kw_batches_ori, permutation_cache, kw_sub_per, kw_sub_temp, sub_model, sub_tokenizer)
        input_ids = my_model.encode(inputs).input_ids
        
        output = my_model.model.generate(
            input_ids=input_ids, 
            do_sample=do_sample,
            num_beams=2*n_beams,
            top_k=top_k,
            top_p=top_p,
            early_stopping = early_stopping,
            temperature=temperature,
            num_return_sequences=n_beams
           )
        output = my_model.tokenizer.batch_decode(output)
        for out in output:
            X_decoded = get_output(X_processed, out)
            X_cands.append(X_decoded)
        
    #if len(X_cands) == 0:
    #    print(output)
    #    print(X_decoded)
    #    input()
    try:
        X_cands_filtered = filter_cands(X, X_cands)
        score_k = min(score_k, len(X_cands_filtered))
        assert len(X_cands_filtered) > 0
    except AssertionError:
        print("No valid candidates")
        X_cands_filtered = X_cands
        score_k = 1
    
    #print(X_cands_filtered)
    #print(len(X_cands))
    #print(X_cands[:score_k])
    text_top_k, sem_rank, ppl_rank, div_rank = Scoring(args, X, X_cands_filtered, score_k, gpt_model, gpt_tokenizer, sem_model, sem_tokenizer)
    
    #generated_s.append(text_top_k[0])
    
    '''
    print("Semantic Ranking:")
    for i, l in enumerate(sem_rank):
        print(('%s. %s' % (str(i+1), l)))
        
    print("PPL Ranking:")
    for i, l in enumerate(ppl_rank):
        print(('%s. %s' % (str(i+1), l)))
        
    print("Diversity Ranking:")
    for i, l in enumerate(div_rank):
        print(('%s. %s' % (str(i+1), l)))
        
    print("All considered:")
    for i, l in enumerate(text_top_k):
        print(('%s. %s' % (str(i+1), l)))
    '''
    print("Output: ")
    print(text_top_k[0])
    
        
    out = { 
        "context": [sample[0],sample[2]],
        "X": X,
        "KW": kw_batches_ori,
        "sem_rank": sem_rank[:5],
        "Y": text_top_k[0]
    }
    json_dump.append(out)
    if (idx+1)%1000 == 0:
        output_file = os.path.join(output_path, 'test1m-{}-{}.json'.format(test_id, int(save_id + (idx+1)/1000 - 1)))
        if if_save:
            with open(output_file, 'w') as f:
                json.dump(json_dump, f)
            json_dump = [vars(args)]

'''
if args.step > 1000:
    end_id = int((args.step+args.start)/1000 - 1)
    output_file = os.path.join(output_path, 'test100k-{}-{}-{}.json'.format(test_id, save_id, end_id))
else:
    output_file = os.path.join(output_path, 'test100k-{}-{}.json'.format(test_id, save_id))
if if_save:
    with open(output_file, 'w') as f:
        json.dump(json_dump, f)
'''
        
'''store_name = '/data/ljx/results/para/beam{}_num{}.pkl'.format(n_beams, test_number)
with open(store_name, 'wb') as f:
    pkl.dump(generated_s,f)

evaluation(actual_s, generated_s, 'bleu')
evaluation(actual_s, generated_s, 'rouge')'''

