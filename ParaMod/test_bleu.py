from fast_bleu import BLEU
from rouge_score import rouge_scorer
from collections import defaultdict

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from termcolor import colored
from tqdm import tqdm
smooth = SmoothingFunction()

def filter_special_tokens(sent):
    sent = sent.strip().lower()
    if '<extra_id_0>' in sent:
        sent = sent.split('<extra_id_0>')[1]
    if '<extra_id_1>' in sent:
        sent = sent.split('<extra_id_1>')[0]
    while sent.endswith('</s>'):
        sent = sent[:-len('</s>')].strip()
    if sent[:1] == '"' or sent[:1] == '.':
        sent = sent[1:].strip()
    while(sent[-2:] == '?.'):
        sent = sent[:-1].strip()
    sent = sent.replace('.', ' .')
    sent = sent.replace('?', ' ?')
    sent = sent.replace('!', ' !')
    sent = ' '.join(sent.replace('<unk>', '').split(' '))
    return sent

def gt_filter(sent):
    sent = sent.strip().lower()
    mark = sent[-1]
    sent = sent[:-1] + ' ' + mark
    return sent

'''
inputX = ['What are the most intellectually stimulating movies you have ever seen?',
         'What are the basic types of satellites?',
         'How can I find my all Gmail accounts?',
         'Is it possible to make time machine and do time travel?']

target = ['What are the most intellectually stimulating films you have ever watched?',
          'What are different types of satellites? What is the most advanced type?',
          'How can I find all my old Gmail accounts?',
          'Is time travel possible in next 5 years?']

inputX = [ input_filter(i) for i in inputX ]
target = [ gt_filter(t) for t in target ]
'''

dataset = "QQP_split"
test_num = 30000

#beamgen_path = '/data/ljx/result/para_model/t5-base-stage-5epochs_2022-06-04/inference/6_4/inferenced_mscoco-t5base-3epoch-sampling-T1.5_top-15-p0.92-T1.5_seed1234.txt'
#beamgen_path = '/data/ljx/result/para_model/t5-base-stage_2022-04-27/inference/6_4/inferenced_qqp-ablation-25epoch_top-15-p0.92-T1.5_seed1234.txt'
#inferenced_mscoco-t5base-sampling-T1.5_top-15-p0.92-T1.5_seed1234.txt'

#beamgen_path = '/data/ljx/result/para_model/t5-base-stage_2022-04-27/inference/inferenced_qqp-t5_base-sampling-T_2_top-10-p1.0-T2.0_seed1234.txt'
#beamgen_path = '/data/ljx/result/para_model/t5-base_ft-2epoch-1000step_continued_2022-05-06/inference/inferenced_QQP-t5base-ft_top-10-p1.0-T1.0_seed1234.txt'
#beamgen_path = '/data/ljx/result/para_model/t5-base_ft-qqp-10k_continued_2022-06-09/inference/6_4/inferenced_qqp-t5base-ft10k_top-15-p0.92-T1.5_seed1234.txt'

#beamgen_path = '/data/ljx/result/para_model/t5-base_common_ft-kqapro-100_continued_2022-06-12/inference/6_4/inferenced_kqapro-t5base-c-ft100_top-15-p0.92-T1.5_seed1234.txt'
#beamgen_path = '/data/ljx/result/para_model/t5-base_ft-mscoco-10k_continued_2022-06-09/inference/6_4/inferenced_mscoco-t5base-ft10k_top-15-p0.92-T1.5_seed1234.txt'
#beamgen_path = '/data/ljx/result/para_model/t5-base-stage_2022-04-27/inference/new_filtered/inferenced_twitter-t5base-stage1-filtered_top-10-p1.0-T1.0_seed1234.txt'
beamgen_path = '/data/ljx/result/para_model/t5-base-abla-25k_2022-06-22/inference/6_4/inferenced_qqp-ablation-25k_top-15-p0.92-T1.5_seed1234.txt'


#corrupted_path = '/home/ljx/unsup_corruption_hedge/results/filtered/inferenced_mscoco-gpt2-10epochs_top-10-p1.0-T1.0_seed1234.txt'
#corrupted_path = '/home/ljx/unsup_corruption_hedge/results/filtered/inferenced_qqp_top-10-p1.0-T1.0_seed1234.txt'
#corrupted_path = '/data/ljx/result/para_model/t5-base_common_ft-kqapro-500_continued_2022-06-12/inference/6_4/inferenced_kqapro-t5base-c-ft500_top-15-p0.92-T1.5_seed1234.txt'
corrupted_path = '/data/ljx/result/para_model/t5-base-abla-50k_2022-06-22/inference/6_4/inferenced_qqp-ablation-50k_top-15-p0.92-T1.5_seed1234.txt'



#sampgen_path = '/data/ljx/result/para_model/t5-base-stage-5epochs_2022-06-04/inference/6_4/inferenced_mscoco-t5base-3epoch-sampling-T2.0_top-20-p0.92-T2.0_seed123456.txt'
#sampgen_path = '/data/ljx/result/para_model/t5-base-stage_2022-04-27/inference/6_4/inferenced_qqp-ablation-20epoch_top-15-p0.92-T1.5_seed1234.txt'
sampgen_path = '/data/ljx/result/para_model/t5_base/inference/6_4/inferenced_qqp-ablation-0k_top-15-p0.92-T1.5_seed1234.txt'

#inferenced_QQP-t5base-sampling-T1.5_top-15-p0.92-T1.5_seed1234.txt'
#ampgen_path = '/data/ljx/result/para_model/t5-base_ft-2epoch-1000step_continued_2022-05-06/inference/6_4/inferenced_QQP-t5base-ft500-sampling-T1.5_top-15-p0.92-T1.5_seed1234.txt'
#sampgen_path = '/data/ljx/result/para_model/t5-base-stage_2022-04-27/inference/inferenced_twitter-t5base-stage1-filtered_top-10-p1.0-T1.0_seed1234.txt'
#sampgen_path = '/data/ljx/result/para_model/t5-base_common_ft-kqapro-1k_continued_2022-06-12/inference/6_4/inferenced_kqapro-t5base-c-ft1k_top-15-p0.92-T1.5_seed1234.txt'
#sampgen_path = '/data/ljx/result/para_model/t5-base-abla-75k_2022-06-22/inference/6_4/inferenced_qqp-ablation-75k_top-15-p0.92-T1.5_seed1234.txt'


target_path = 'data/%s/test_target.txt' % dataset
input_path = 'data/%s/test_input.txt' % dataset

weights = (1/4., 1/4., 1/4., 1/4.)#{'4gram':(1/4., 1/4., 1/4., 1/4.)}

input_file = open(input_path, 'r')
target_file = open(target_path, 'r')

beamgen_file = open(beamgen_path, 'r')
corrgen_file = open(corrupted_path, 'r')
sampgen_file = open(sampgen_path, 'r')

beam_cache = ""
corr_cache = ""
samp_cache = ""

scores_beamgen = defaultdict(list)
scores_corrgen = defaultdict(list)
scores_sampgen = defaultdict(list)

rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2'], use_stemmer=True)

for row_ind in tqdm(range(test_num)):
    Input = input_file.readline()
    Target = target_file.readline()
    
    Input = filter_special_tokens(Input)
    Target = filter_special_tokens(Target)
    
    beamgen_sents = []
    corrgen_sents = []
    sampgen_sents = []
    if len(beam_cache) > 0 and int(beam_cache[0]) == row_ind:
        beamgen_sents.append(beam_cache[1])
    if len(corr_cache) > 0 and int(corr_cache[0]) == row_ind:
        corrgen_sents.append(corr_cache[1])
    if len(samp_cache) > 0 and int(samp_cache[0]) == row_ind:
        sampgen_sents.append(samp_cache[1])
    
    while(True):
        beam_cache = beamgen_file.readline()
        if len(beam_cache) == 0:
            break
        beam_cache = beam_cache.split('\t')
        ind = int(beam_cache[0])
        sen = beam_cache[1]
        if ind!=row_ind:
            break
        beamgen_sents.append(sen)
    
    while(True):
        corr_cache = corrgen_file.readline()
        if len(corr_cache) == 0:
            break
        corr_cache = corr_cache.split('\t')
        ind = int(corr_cache[0])
        sen = corr_cache[1]
        if ind!=row_ind:
            break
        corrgen_sents.append(sen)
        
    while(True):
        samp_cache = sampgen_file.readline()
        if len(samp_cache) == 0:
            break
        samp_cache = samp_cache.split('\t')
        ind = int(samp_cache[0])
        sen = samp_cache[1]
        if ind!=row_ind:
            break
        sampgen_sents.append(sen)
    
    for sen in beamgen_sents:
        sen = filter_special_tokens(sen)
        #gt_bleu = BLEU([Target.split()])
        #ip_bleu = BLEU([Input.split()])
        #bs = gt_bleu.get_score([sen.split()])[4][0]
        #tmp = ip_bleu.get_score([sen.split()])[4][0]
        bs = sentence_bleu([Target.split()], sen.split(), smoothing_function=smooth.method1, weights=weights)
        tmp = sentence_bleu([Input.split()], sen.split(), smoothing_function=smooth.method1, weights=weights)
        ibs = 0.8*bs - 0.2*tmp
        rs = rouge.score(Target, sen)
        scores_beamgen[row_ind].append([ibs, bs, rs['rouge1'].fmeasure, rs['rouge2'].fmeasure, sen])
    for sen in corrgen_sents:
        sen = filter_special_tokens(sen)
        #gt_bleu = BLEU([Target.split()])
        #ip_bleu = BLEU([Input.split()])
        #bs = gt_bleu.get_score([sen.split()])[4][0]
        #tmp = ip_bleu.get_score([sen.split()])[4][0]
        bs = sentence_bleu([Target.split()], sen.split(), smoothing_function=smooth.method1, weights=weights)
        tmp = sentence_bleu([Input.split()], sen.split(), smoothing_function=smooth.method1, weights=weights)
        ibs = 0.8*bs - 0.2*tmp
        rs = rouge.score(Target, sen)
        scores_corrgen[row_ind].append([ibs, bs, rs['rouge1'].fmeasure, rs['rouge2'].fmeasure, sen])
    
    for sen in sampgen_sents:
        sen = filter_special_tokens(sen)
        #gt_bleu = BLEU([Target.split()])
        #ip_bleu = BLEU([Input.split()])
        #bs = gt_bleu.get_score([sen.split()])[4][0]
        #tmp = ip_bleu.get_score([sen.split()])[4][0]
        bs = sentence_bleu([Target.split()], sen.split(), smoothing_function=smooth.method1, weights=weights)
        tmp = sentence_bleu([Input.split()], sen.split(), smoothing_function=smooth.method1, weights=weights)
        ibs = 0.8*bs - 0.2*tmp
        rs = rouge.score(Target, sen)
        scores_sampgen[row_ind].append([ibs, bs, rs['rouge1'].fmeasure, rs['rouge2'].fmeasure, sen])
        
    scores_beamgen[row_ind].sort(key=lambda x: x[0], reverse=True)
    scores_corrgen[row_ind].sort(key=lambda x: x[0], reverse=True)
    scores_sampgen[row_ind].sort(key=lambda x: x[0], reverse=True)
    
    '''
    print(colored("Input: %s" % Input, 'yellow'))
    print(colored("Target: %s" % Target, 'yellow'))
    print("beam gen result:")
    for i in range(min(3, len(beamgen_sents))):
        print(scores_beamgen[row_ind][i][4], colored(str(round(scores_beamgen[row_ind][i][0], 2)), 'magenta'), 
              round(scores_beamgen[row_ind][i][1], 2), round(scores_beamgen[row_ind][i][2], 2), 
              round(scores_beamgen[row_ind][i][3], 2))
    print("corr gen result:")
    for i in range(min(3, len(corrgen_sents))):
        print(scores_corrgen[row_ind][i][4], colored(str(round(scores_corrgen[row_ind][i][0], 2)), 'magenta'), 
              round(scores_corrgen[row_ind][i][1], 2), round(scores_corrgen[row_ind][i][2], 2), 
              round(scores_corrgen[row_ind][i][3], 2))
    print("samp gen result:")
    for i in range(min(3, len(sampgen_sents))):
        print(scores_sampgen[row_ind][i][4], colored(str(round(scores_sampgen[row_ind][i][0], 2)), 'magenta'), 
              round(scores_sampgen[row_ind][i][1], 2), round(scores_sampgen[row_ind][i][2], 2), 
              round(scores_sampgen[row_ind][i][3], 2))
    print('\n')
    input()
    '''
    
    
input_file.close()
target_file.close()
beamgen_file.close()
corrgen_file.close()
sampgen_file.close()
print(len(scores_beamgen))
print(len(scores_corrgen))
print(len(scores_sampgen))

print("Average score:")
print("Beam gen:")
for key in scores_beamgen.keys():
    scores_beamgen[key].sort(key=lambda x: x[0], reverse=True)
avg = sum( [score[0][0] for score in scores_beamgen.values()] )/len(scores_beamgen)
print("iBLEU:", colored(str(round(avg,4)), 'red'))

#for key in scores_beamgen.keys():
#    scores_beamgen[key].sort(key=lambda x: x[1], reverse=True)
avg = sum( [score[0][1] for score in scores_beamgen.values()] )/len(scores_beamgen)
print("BLEU:", colored(str(round(avg,4)), 'red'))

#for key in scores_beamgen.keys():
#    scores_beamgen[key].sort(key=lambda x: x[2], reverse=True)
avg = sum( [score[0][2] for score in scores_beamgen.values()] )/len(scores_beamgen)
print("R1:", colored(str(round(avg,4)), 'red'))

#for key in scores_beamgen.keys():
#    scores_beamgen[key].sort(key=lambda x: x[3], reverse=True)
avg = sum( [score[0][3] for score in scores_beamgen.values()] )/len(scores_beamgen)
print("R2:", colored(str(round(avg,4)), 'red'))


print("Corr gen:")
for key in scores_corrgen.keys():
    scores_corrgen[key].sort(key=lambda x: x[0], reverse=True)
avg = sum( [score[0][0] for score in scores_corrgen.values() if len(score)>0] )/len(scores_corrgen)
print("iBLEU:", colored(str(round(avg,4)), 'red'))

#for key in scores_corrgen.keys():
#    scores_corrgen[key].sort(key=lambda x: x[1], reverse=True)
avg = sum( [score[0][1] for score in scores_corrgen.values() if len(score)>0] )/len(scores_corrgen)
print("BLEU:", colored(str(round(avg,4)), 'red'))

#for key in scores_corrgen.keys():
#    scores_corrgen[key].sort(key=lambda x: x[2], reverse=True)
avg = sum( [score[0][2] for score in scores_corrgen.values() if len(score)>0] )/len(scores_corrgen)
print("R1:", colored(str(round(avg,4)), 'red'))


#for key in scores_corrgen.keys():
#    scores_corrgen[key].sort(key=lambda x: x[3], reverse=True)
avg = sum( [score[0][3] for score in scores_corrgen.values() if len(score)>0] )/len(scores_corrgen)
print("R2:", colored(str(round(avg,4)), 'red'))


print("Samp gen:")
for key in scores_sampgen.keys():
    scores_sampgen[key].sort(key=lambda x: x[0], reverse=True)
avg = sum( [score[0][0] for score in scores_sampgen.values()] )/len(scores_sampgen)
print("iBLEU:", colored(str(round(avg,4)), 'red'))

#for key in scores_sampgen.keys():
#    scores_sampgen[key].sort(key=lambda x: x[1], reverse=True)
avg = sum( [score[0][1] for score in scores_sampgen.values()] )/len(scores_sampgen)
print("BLEU:", colored(str(round(avg,4)), 'red'))

#for key in scores_sampgen.keys():
#    scores_sampgen[key].sort(key=lambda x: x[2], reverse=True)
avg = sum( [score[0][2] for score in scores_sampgen.values()] )/len(scores_sampgen)
print("R1:", colored(str(round(avg,4)), 'red'))

#for key in scores_sampgen.keys():
#    scores_sampgen[key].sort(key=lambda x: x[3], reverse=True)
avg = sum( [score[0][3] for score in scores_sampgen.values()] )/len(scores_sampgen)
print("R2:", colored(str(round(avg,4)), 'red'))

'''
#rouge1_scores = defaultdict(list)
#rouge2_scores = defaultdict(list)
nltk_bleu_scores = defaultdict(list)

rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2'], use_stemmer=True)
            
for i in range(len(inputX)):
    print("Input:", inputX[i])
    print("Target:", target[i])
    gt_bleu = BLEU([target[i].split()])
    for j,sen in enumerate(generated[i]):
        print("Cand {}:".format(j), sen)
        scores = rouge.score(target[i], sen)
        nltk_bs = sentence_bleu([target[i].split()], sen.split(), smoothing_function=smooth.method1)
        nltk_bleu_scores[i].append(
            (nltk_bs, scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, sen))
    hyp = [ sen.split() for sen in generated[i] ]
    #hyp.append("What is this is this a test dummy sentence ?".split())
    res = gt_bleu.get_score(hyp)
    print(res)
    print(nltk_bleu_scores[i])
    print('\n')

    # 后处理： <unk>, 最后的标点"?.", 大小写
'''