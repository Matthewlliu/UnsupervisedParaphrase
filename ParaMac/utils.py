import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet as wd
from nltk.corpus import stopwords
from nltk.translate.bleu_score import corpus_bleu

from rake_nltk import Rake
import bert_score
from bert_scorer import score as SCORE
from rouge import Rouge
from random import shuffle
import numpy as np
from keybert import KeyBERT

from ppl_with_gpt import ppl_score

import copy
import logging
import torch
import transformers
import json
import pickle
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

TAGS = ['FW', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBG', 'VBN', 'VBD', 'VBP', 'VBZ']
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
#'PRP'

bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}

split_token = '<fun_spt>'
arg_token = '<fun_arg>'

def end_process(text, ori_len):
    text = text.strip().split('\n')
    text = ' '.join(text)
    gen_len = len(text) - ori_len
    ori_text = text[:ori_len]
    gen_text = text[ori_len:]
    if gen_text[-1]!='.':
        gen_text = gen_text.split('.')
        if sum([len(gt) for gt in gen_text[:-1]]) > gen_len*2/3:
            gen_text = '.'.join(gen_text[:-1]) + '.'
        else:
            gen_text = '.'.join(gen_text) + '.'
    text = ori_text + gen_text
    return text

def new_end_process(text):
    text = text.strip().split('\n')
    text = ' '.join(text)
    template = '<extra_id'
    new_text = text.split(template)[0]
    ori_len = len(new_text)
    try:
        if new_text[-1]!='.':
            new_text = new_text.split('.')
            if sum([len(gt) for gt in new_text[:-1]]) > ori_len*2/3:
                new_text = '.'.join(new_text[:-1]) + '.'
            else:
                new_text = '.'.join(new_text) + '.'
    except IndexError:
        with open('end_error.txt', 'a') as f:
            f.write(text + '\n')
        return text
    return new_text

def kw_process(kw, tokenizer, if_bart=False):
    #kw_candidate = []#[ tokenizer.convert_tokens_to_ids(k) for k in kw ]
    sen = 'i ' + ' '.join(kw)
    tmp = tokenizer.encode(sen, return_tensors='pt')
    if if_bart:
        tmp = tmp[0][2:-1]
    else:
        tmp = tmp[0][1:]
    #print(tmp)
    kw_candidate = [kw for kw in tmp if kw!=50256]
    #kw_candidate = list( set(kw_candidate) - set([50256]) )
    return kw_candidate

def kw_extraction_expand(sentence):
    words = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    #print()
    
    ret = []
    for word, pos in pos_tags:
        if pos not in TAGS:
            continue
        
        if pos in ['FW', 'MD']:
            ret.append([word])
            continue
        elif pos[:2] == 'JJ':
            a = ['a', 's']
        elif pos[:2] == 'NN':
            a = ['n']
        elif pos[:2] == 'RB':
            a = ['r']
        elif pos[:2] == 'VB':
            a = ['v']
        tmp = []
        syns = wd.synsets(word)
        for ss in syns:
            name = ss.name().split('.')[1]
            if name not in a:
                continue
            tmp.extend(ss.lemma_names())
        if len(tmp)>0:
            tmp.append(word)
            tt = []
            for ww in tmp:
                www = ww.split('_')
                for ss in www:
                    if len(ss) > 0:
                        tt.append(ss)
            tmp = tt
            ret.append(list(set(tmp)))
    return ret

def kw_pos_extraction(sentence):
    ret = []
    words = word_tokenize(sentence)
    if "n't" in words:
        ret.append("n't")
    pos_tags = nltk.pos_tag(words)
    for word, pos in pos_tags:
        if pos in NOUNS or pos in VERBS:
            ret.append(word)
    return ret

def kw_extraction(sentence):
    marks = [',', '.', '!', '?', '"', '(', ':', '-']
    # strange long '-'
    if ord(sentence[0]) == 8212:
        sentence = sentence[1:]
        
    keywords = kw_rake(sentence)
    #print(keywords)
    #keywords = copy.deepcopy(kw_r)
    #print(keywords)
    # check the '-' word
    for k in sentence.split(' '):
        if '-' in k:
            if k[-1] in marks:
                k = k[:-1].strip()
            elif k[0] in marks:
                k = k[1:].strip()
                
            # check if the word in already in kw in one piece
            flag = False
            for kw in keywords:
                if k in kw:
                    flag = True
            
            if flag is False:
                parts = k.split('-')
                tmp = copy.deepcopy(keywords)
                for kw in keywords:
                    if parts[0] == kw.split(' ')[-1]:
                        parts[0] = kw
                        tmp = [ kk for kk in tmp if kk!=kw ]
                    elif parts[-1] == kw.split(' ')[0]:
                        parts[-1] = kw
                        tmp = [ kk for kk in tmp if kk!=kw ]

                tmp.append( '-'.join(parts) )
                keywords = copy.deepcopy(tmp)
    tmp = []
    
    for p in keywords:
        tmp.extend(p.split(' '))
        
    kw_p = kw_pos_extraction(sentence)
    sw = stopwords.words('english')
    
    for k in kw_p:
        if (k not in tmp) and (k not in sw):
            keywords.append(k)
            
    marks = [',', '.', '!', '?', '"', '(', ':', '-','—']
    #tmp = []
    for i in range(len(keywords)):
        try:
            while(keywords[i][-1] in marks):
                keywords[i] = keywords[i][:-1].strip()
            while(keywords[i][0] in marks):
                keywords[i] = keywords[i][1:].strip()
        except IndexError:
            continue
    keywords = list(set(keywords))
    keywords = [ k for k in keywords if len(k)>0 ]
    return keywords

def kw_rake(sentence):
    r = Rake()
    r.extract_keywords_from_text(sentence)
    return r.get_ranked_phrases()

def kw_keybert(sentence):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 3), stop_words=None)

    return keywords
    

def Scoring(args, X, X_cands, k, gpt_model, gpt_tokenizer, sem_model, sem_tokenizer):
    X_list = [ X for _ in range(len(X_cands))]
    _, _, semantic_score = SCORE(X_cands, X_list, model=sem_model, tokenizer=sem_tokenizer, lang='en')
    #semantic_score = F1
    
    
    #code for ppl evaluation
    fluency_score = ppl_score(X_cands, gpt_model, gpt_tokenizer)
    # diversity
    diversity_score = div_score(X, X_cands)
    
    #print("semantic score:", np.mean(semantic_score.numpy()), '±', np.std(semantic_score.numpy()))
    #print("ppl score:", np.mean(fluency_score), '±', np.std(fluency_score))
    #print("diversity score:", np.mean(diversity_score), '±', np.std(diversity_score))

    final_score = args.sem_weight*semantic_score + args.ppl_weight*fluency_score + args.div_weight*diversity_score
    topk = final_score.topk(k).indices
    ret = []
    for idx in topk:
        ret.append(X_cands[idx])
        
    fluency_score = torch.tensor(fluency_score)
    diversity_score = torch.tensor(diversity_score)
    
    topk = semantic_score.topk(k).indices
    sem_rank = []
    for idx in topk:
        sem_rank.append(X_cands[idx])
    topk = fluency_score.topk(k).indices
    ppl_rank = []
    for idx in topk:
        ppl_rank.append(X_cands[idx])
    topk = diversity_score.topk(k).indices
    div_rank = []
    for idx in topk:
        div_rank.append(X_cands[idx])
    return ret, sem_rank, ppl_rank, div_rank

def div_score(X, X_cands):
    alpha = 0.5
    beta = 0.5
    
    X_gram = set(X.strip().split(' '))
    X_gram_list = X.strip().split(' ')
    X_len = len(X_gram)
    score = []
    for cand in X_cands:
        cand_gram = set(cand.strip().split(' '))
        cand_gram_list = cand.strip().split(' ')
        cand_len = len(cand_gram)
        
        intersection = X_gram.intersection(cand_gram)
        union = X_gram.union(cand_gram)
        part1 = alpha * ( len(intersection) / len(union) )
        
        div = max(len(X_gram), len(cand_gram))
        if len(intersection) != 0:
            part2 = beta * sum( [np.abs(X_gram_list.index(inter) - cand_gram_list.index(inter))/div for inter in intersection] ) / len(intersection)
        else:
            part2 = 0
        
        score.append(part1 + part2)
    return np.array(score) * 2
        

def evaluation(actual_sentence, generated_sentence, mode):
    actual_words_list = []
    for line in actual_sentence:
        actual_words_list.append(line.strip().lower().split())
    generated_words_list = []
    for line in generated_sentence:
        generated_words_list.append(line.strip().lower().split())
    
    if mode == 'bleu':
        evaluate_bleu(actual_words_list, generated_words_list)
    elif mode =='rouge':
        evaluate_rouge(actual_sentence, generated_sentence)
    else:
        print('not a valid argument: mode')

def get_corpus_bleu_scores(actual_word_lists, generated_word_lists):
    bleu_scores = dict()
    for i in range(len(bleu_score_weights)):
        bleu_scores[i + 1] = round(
            corpus_bleu(
                list_of_references=actual_word_lists[:len(generated_word_lists)],
                hypotheses=generated_word_lists,
                weights=bleu_score_weights[i + 1]), 4)
    return bleu_scores

def evaluate_bleu(actual_words_list, generated_words_list):
    bleu_scores = get_corpus_bleu_scores(actual_words_list, generated_words_list)
    sumss = 0
    for s in bleu_scores:
        sumss += 0.25*bleu_scores[s]
    print('bleu scores:', sumss, bleu_scores)

def evaluate_rouge(actual_words_list, generated_words_list):
    rouge = Rouge()
    scores = rouge.get_scores(generated_words_list, actual_words_list, avg=True)
    print('Rouge score:', scores)


def get_input(sentence_list, mid_id, kw_batches, permutation_cache, kw_sub_per, kw_sub_temp, sub_model, sub_tokenizer=None):
    template = "<extra_id_{}>"
    pre_text = ' '.join(sentence_list[:mid_id])
    pro_text = ' '.join(sentence_list[mid_id+1:])
    X = sentence_list[mid_id]
    
    # shuffle kw
    while(kw_batches in permutation_cache):
        shuffle(kw_batches)
    permutation_cache.append(copy.deepcopy(kw_batches))
    
    #shuffle(kw_batches)
    #while(kw_batches == origin_kw):
    #    shuffle(kw_batches)
    #print(kw_batches)
    
    # substitute part of the key words
    #kw_batches = kw_substi(sentence_list, mid_id, kw_batches, kw_sub_per, kw_sub_temp, sub_model)
    kw_batches = kw_substi_bert(sentence_list, mid_id, kw_batches, kw_sub_per, kw_sub_temp, sub_model, sub_tokenizer)
    #print(kw_batches)
    #print('\n')
     
    tmp = [template.format(0)]
    for i in range(len(kw_batches)):
        tmp.append(kw_batches[i])
        tmp.append(template.format(i + 1))
    X_processed = ' '.join(tmp)
    inputs = pre_text + ' ' + X_processed + ' ' + pro_text
    #print(inputs)
    #input()
    return inputs, X_processed, permutation_cache
    
def get_output(X_processed, output):
    """
        X_processed: 1 sentence with masks
        output: mask predictions
        
        Y: 3 sentences
        X_decoded: 1 sentence filled with predictions
    """
    template = "<extra_id_{}>"
    i = 0
    while(True):
        r = output.split(template.format(i))
        if len(r) == 1:
            break
        r = r[1]
        d = r.split(template.format(i+1))[0].strip()
        X_processed = X_processed.replace(template.format(i), d)
        i += 1
    X_decoded = new_end_process(X_processed)
    return X_decoded

def kw_sort(sentence, kws):
    kw_dict = {}
    for kw in kws:
        text = sentence.split(kw)[0]
        kw_dict[kw] = len(text)
    tmp = sorted(kw_dict.items(), key=lambda kv:(kv[1], kv[0]))
    kw_copy = []
    for item in tmp:
        kw_copy.append(item[0])
    return kw_copy

def kw_filter(sentence, kw_batches, kw_drop_per, my_model):
    # filter kw
    # only to filter if there are more than one key word
    if kw_drop_per>0 and len(kw_batches)>1:
        labels = my_model.tokenizer(sentence, return_tensors='pt').input_ids.cuda()
        loss = dict()
        for kw in kw_batches:
            input_ids = my_model.tokenizer(kw, return_tensors='pt').input_ids.cuda()
            with torch.no_grad():
                l = my_model.model(input_ids=input_ids, labels=labels).loss
            loss[kw] = l
        kw_batches = sorted(loss.keys(), key=lambda k:loss[k])
        #import math
        drop_num = round(kw_drop_per * len(kw_batches))
        if drop_num > 0:
            kw_batches = kw_batches[:-drop_num]
    return kw_batches

def kw_substi(sentence, mid, kw_batches, kw_sub_per, kw_sub_temp, my_model):
    kw_seq = '|||'.join(kw_batches)
    kw_sub_num = round( len(kw_batches)*kw_sub_per )
    kw_to_subs = np.random.choice(kw_batches, kw_sub_num, replace=False)
    #tmp = []
    #for kw in kw_to_subs:
    #    tmp.extend(kw.split(' '))
    #kw_to_subs = tmp
    X = sentence[mid]
    
    for kw_to_sub in kw_to_subs:
        tmp = X.split(kw_to_sub)
        try:
            inputs = ' '.join([sentence[mid-1], tmp[0], '<extra_id_0>', tmp[1], sentence[mid+1]])
        except IndexError:
            print("key word insplitable: extra blank space")
            print(kw_to_sub)
            print(tmp)
            continue

        #print("Inputs:", inputs)
        input_ids = my_model.tokenizer(inputs, return_tensors='pt').input_ids.cuda()
        output = my_model.model.generate(
            input_ids=input_ids,
            do_sample=True,
            num_beams=2,
            top_k=50,
            temperature=kw_sub_temp,
            num_return_sequences=2
        )
        output = my_model.tokenizer.batch_decode(output)
        #print(kw_to_sub)
        #print(output)
        output = output[1].split('<extra_id_0>')
        try:
            output = output[1].strip().split('<extra_id_1>')
        except IndexError:
            continue
        output = output[0].strip()
        #print(output)
        #print('\n')
        #input()

        if check_validation(kw_to_sub, output):
            kw_seq = kw_seq.split(kw_to_sub)
            try:
                kw_seq = kw_seq[0] + output + kw_seq[1]
            except IndexError:
                #print(kw_to_sub)
                #print(kw_seq)
                kw_seq = kw_seq[0]
        #print(kw_seq)
    return kw_seq.split('|||')

def kw_substi_bert(sentence, mid, kw_batches, kw_sub_per, kw_sub_temp, sub_model, sub_tokenizer):
    # get position and part-of-speech
    pos_dict = dict()
    X = sentence[mid]
    kw_sub_num = round( len(kw_batches)*kw_sub_per )
    kw_to_subs = np.random.choice(kw_batches, kw_sub_num, replace=False)
    
    for kw in kw_to_subs:
        words = kw.split(' ')
        fir = X.split(words[0])[0]
        fin = X.split(words[-1])[0]
        fir = len(fir.split())
        fin = len(fin.split())+1
        pos_dict[kw] = [fir, fin]
    #for v in pos_dict.values():
    #    for i in range(v[0], v[1]):
    #        print(X.split()[i])
    words = word_tokenize(X)
    pos_tags = nltk.pos_tag(words)
    POS = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    pos_words = {}
    for k in pos_tags:
        if k[1] in POS:
            pos_words[k[0]] = k[1]
    #print(pos_words)
    
    marks = [',', '.', '!', '?', '"', '(', ':', '-','—']
    kw_words = {}
    for kw in kw_to_subs:
        tmp = kw.split(' ')
        for i in range(len(tmp)):
            k = tmp[i]
            if len(k)>0:
                while( k[-1] in marks):
                    k = k[:-1].strip()
                    if len(k)==0:
                        break
            if len(k)>0:
                while( k[0] in marks):
                    k = k[1:].strip()
                    if len(k)==0:
                        break
            tmp[i] = k
            
        tmp = [k for k in tmp if len(k)>0]
        kw_words[kw] = copy.deepcopy(tmp)
    #print(kw_words.keys())
    
    to_ret = [ kw for kw in kw_batches if kw not in kw_words]
    for k, v in kw_words.items():
        word_tmp = k
        for word in v:
            if word in pos_words:
                tmp = X.split(word)
                inputs = ' '.join([sentence[mid-1], tmp[0], '[MASK]', tmp[1], sentence[mid+1]])
                #print(inputs)
                # predict
                inputs = sub_tokenizer(inputs.strip(), return_tensors="pt").to('cuda')
                try:
                    with torch.no_grad():
                        logits = sub_model(**inputs, return_dict=True).logits
                except RuntimeError:
                    continue
                mask_token_index = (inputs.input_ids == sub_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
                outs = sub_tokenizer.decode(predicted_token_id)
                flag = True
                if pos_words[word] in POS:#['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                    flag = check_valid(word, outs)
                if flag:
                    word_tmp = word_tmp.replace(word, outs)
        to_ret.append(word_tmp)
    #print(to_ret)
    return to_ret
    
def check_valid(word, outs):
    syns = wd.synsets(word)
    syns_list = [ ss.lemma_names() for ss in syns ]
    tmp = []
    for s in syns_list:
        tmp.extend(s)
    syns_list = tmp
    if outs in syns_list:
        return True
    else:
        return False
    
def check_validation(kw_to_sub, output):
    """
        can only check the circumstance that the phrase is made up of an ADJ plus a NOUN
    """
    if kw_to_sub == output:
        return False
    kw_words = word_tokenize(kw_to_sub)
    out_words = word_tokenize(output)
    # if the length is different, it's hard to align the syn. or asyn.
    #print(kw_words)
    #print(out_words)
    if len(kw_words) != len(kw_words):
        return True
    
    if len(kw_words) == 1:
        #print('1')
        pos = nltk.pos_tag(kw_words)
        if pos[0][1] in ['JJ', 'JJR', 'JJS']:
            syns = wd.synsets(kw_words[0])
            syns_list = [ ss.lemma_names() for ss in syns ]
            tmp = []
            for s in syns_list:
                tmp.extend(s)
            syns_list = tmp
            if output in syns_list:
                return True
            else:
                return False
    elif len(kw_words) == 2:
        #print('2')
        pos = nltk.pos_tag(kw_words)
        if pos[0][1] in ['JJ', 'JJR', 'JJS'] and pos[1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            syns = wd.synsets(kw_words[0])
            syns_list = [ ss.lemma_names() for ss in syns ]
            tmp = []
            for s in syns_list:
                tmp.extend(s)
            syns_list = tmp
            if out_words[0] in syns_list:
                return True
            else:
                return False
        
    return False

def filter_cands(X, X_cands):
    ret = []
    tmp_X = ''.join(X.split(' '))
    for cand in X_cands:
        tmp_cand = ''.join(cand.split(' '))
        if tmp_X !=tmp_cand and len(cand) > 15:
            ret.append(cand)
    return ret

if __name__ == '__main__':
    '''
    cands = ['my heart was racing so fast that it might explode right out of my chest .',
            'my heart was racing so fast that it might explode right out of my chest .']
    refs = ['I could feel that my heart was beating incredibly fast, and it almost felt like exploding in my chest. ', 
           'I just love batman so much from my heart and soul']
    
    sen = cands[0]
    kw = kw_rake(sen)
    print(kw)
    from random import shuffle
    print(shuffle(kw))
    print(kw)
    print(shuffle(kw))
    print(kw)
    exit()
    
    _, _, F1 = SCORE(cands, refs, lang='en', verbose=True)
    print(F1)
    print(np.mean(F1.numpy()))
    
    print(F1.topk(1).indices)
    input_file = '/data/ljx/data/book_corpus/book_corpus_forward.jsonl'
    output_path = '/home/ljx/Megatron-LM-main/tests/'
    samples = sample_ABCD(input_file)
    
    output = '<pad> <extra_id_0> We<extra_id_1> and a movie<extra_id_2>. I went to bed<extra_id_3> slept<extra_id_4> with'
    X_processed = '<extra_id_0> had dinner <extra_id_1> together <extra_id_2> and <extra_id_3>'
    sample = None
    get_output(sample, X_processed, output)
    '''
    #cands = ['my ears heard nothing and i didnt see any movement but my hands shook slightly as i typed , nothing , im fine .', 'i didnt open his reply because i knew what it would say .', 'i already knew that it was strangely quiet .', 'maybe i could stay home and sleep one night .', 'or maybe i could finish a painting tonight and watch mom do origami .', 'i turned the corner , and my house came into view .', 'i started to climb the hill when i froze .']
    #cands = ' '.join(cands)
    
    x = '—come on, champ. i have coveralls in the trunk just for you.'
    y = kw_extraction(x)
    print(y)
  
    '''
    X_processed = 'mr. darmadi quipped that young men , at least in paint , could still capture'
    X_decoded = new_end_process(X_processed)
    print("Final: ",X_decoded)
    '''
    
    