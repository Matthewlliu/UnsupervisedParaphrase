import os
import csv
import argparse
import copy

from model.t5_paraphrase_model import t5_model
from model.gpt2_paraphrase_model import gpt2_model
from model.naive_paraphrase_model import naive_model

class ParaMod():
    def __init__(self, args):
        self.args = args
        self.lm = t5_model(args)
        self.lm.build_model(checkpoint_dir=args.checkpoint_dir)
        
    def generate(self, input_texts, max_length=256, decoding='beam_gen', suffix='', isfilter=True):
        return self.lm.generate_text(input_texts, max_length, decoding, suffix, isfilter)
    

class CorruptLM():
    def __init__(self, args):
        self.args = args
        self.lm = gpt2_model(args)
        self.lm.build_model(checkpoint_dir=args.checkpoint_dir)
        
    def generate(self, input_texts, max_length=256, decoding='sampling', suffix='[SEP]', isfilter=True):
        return self.lm.generate_text(input_texts, max_length, decoding, suffix, isfilter)

def BT(args):
    pass

class Naive():
    def __init__(self, args):
        self.args = args
        self.lm = naive_model(args)
        self.lm.build_model(checkpoint_dir=args.checkpoint_dir)
        
    def generate(self, input_texts):
        return self.lm.Paraphrase(input_texts)
    
def Augment(method, dataset, input_file, output_file):
    print("Read input file %s ..." % input_file)
    
    datain = []
    with open(input_file) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for row in read_tsv:
            datain.append(row)
    if dataset in ['sst-5']:
        header = None
    else:
        header = datain[0]
        datain = datain[1:]
    
    dataout = []
    for row in datain:
        if dataset == 'SST-2':
            sen = row[0]
            label = row[1]
            aug = method.generate([sen])[0]
            
            dataout.append([sen, label])
            for a in aug:
                dataout.append([a, label])
        elif dataset == 'sst-5':
            label = row[0][0]
            sen = row[0][2:]
            aug = method.generate([sen])[0]
            
            dataout.append([label, sen])
            for a in aug:
                dataout.append([label, a])
        elif dataset == 'MNLI':
            sen1 = row[8]
            sen2 = row[9]
            aug = method.generate([sen1, sen2])
            
            dataout.append(copy.deepcopy(row))
            for pair in zip(aug[0], aug[1]):
                tmp = copy.deepcopy(row)
                tmp[8] = sen1
                tmp[9] = pair[1]
                dataout.append(tmp)
        elif dataset == 'SNLI':
            sen1 = row[7]
            sen2 = row[8]
            aug = method.generate([sen1, sen2])
            
            dataout.append(copy.deepcopy(row))
            for pair in zip(aug[0], aug[1]):
                tmp = copy.deepcopy(row)
                tmp[7] = sen1
                tmp[8] = pair[1]
                dataout.append(tmp)
        elif dataset == 'MRPC':
            try:
                sen1 = row[3]
                sen2 = row[4]
            except IndexError:
                parts = row[3].split('\t')
                sen1 = parts[0]
                sen2 = parts[1]
                row[3] = sen1
                row.append(sen2)
            aug = method.generate([sen1, sen2])
            
            dataout.append(copy.deepcopy(row))
            for pair in zip(aug[0], aug[1]):
                tmp = copy.deepcopy(row)
                tmp[3] = sen1
                tmp[4] = pair[1]
                dataout.append(tmp)
        elif dataset == 'QQP':
            sen1 = row[3]
            sen2 = row[4]
            aug = method.generate([sen1, sen2])
            
            dataout.append(copy.deepcopy(row))
            for pair in zip(aug[0], aug[1]):
                tmp = copy.deepcopy(row)
                tmp[3] = sen1
                tmp[4] = pair[1]
                dataout.append(tmp)
        else:
            raise ValueError('Dataset Not Implemented.')
            
    with open(output_file, 'wt') as f:
        if dataset in ['sst-5']:
            writer = csv.writer(f)
        else:
            writer = csv.writer(f, delimiter='\t')
        if header is not None:
            writer.writerow(header)
        for entry in dataout:
            writer.writerow(entry)

def main(args):
    if args.method == 'paramod':
        args.model = 't5-base'
        args.num_generate = args.augment_size - 1
        #args.checkpoint_dir = '/data/ljx/result/para_model/t5-base_ft-mscoco-10k_continued_2022-06-09/checkpoints'
        args.checkpoint_dir = '/data/ljx/result/para_model/t5-base-stage_2022-04-27/checkpoints/checkpoint-31250'
        method = ParaMod(args)
    elif args.method == 'corruption':
        args.model = 'gpt2'
        args.num_generate = args.augment_size - 1
        args.checkpoint_dir = '/data/ljx/result/para_model/gpt2_mscoco-gpt2-10epochs_2022-05-25/checkpoints/checkpoint-30000'
        method = CorruptLM(args)
    elif args.method == 'bt':
        method = BT(args)
    elif args.method == 'naive':
        args.model = 'bert-base-uncased'
        args.num_generate = args.augment_size - 1
        args.checkpoint_dir = '/data/MODELS/%s' % args.model
        method = Naive(args)
    else:
        raise ValueError('Method Not Implemented')
        
    for dataset in args.datasets:
        input_path = os.path.join(args.root, dataset)
        output_path = os.path.join(input_path, args.method)
        #subfolders = ['-'.join([args.origin_K, seed]) for seed in args.seeds]
        for seed in args.seeds:
            source_path = os.path.join(input_path, '-'.join([args.origin_K, seed]))
            target_path = os.path.join(output_path, '-'.join([str(int(args.origin_K)*args.augment_size), seed, args.tag]))
            if not os.path.exists(target_path):
                os.makedirs(target_path)
                
            if dataset in ['sst-5']:
                suffix = 'csv'
            else:
                suffix = 'tsv'
            cmd = 'cp ' + os.path.join(source_path, 'test*%s'%suffix) + ' ' + target_path
            os.system(cmd)
            cmd = 'cp ' + os.path.join(source_path, 'dev*%s'%suffix) + ' ' + target_path
            os.system(cmd)
            source_train = os.path.join(source_path, 'train.%s'%suffix)
            target_train = os.path.join(target_path, 'train.%s'%suffix)
            Augment(method, dataset, source_train, target_train)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,
                        default='paramod',
                        help='Dataset file to paraphrase')
    parser.add_argument('--tag', type=str,
                        default='616',
                        help='')
    parser.add_argument('--augment_size', type=int,
                        default=5,
                        help='Dataset file to paraphrase')
    
    parser.add_argument('--datasets', type=str,
                        default='SST-2,sst-5,MNLI,SNLI,MRPC', #SST-2,sst-5,MNLI,SNLI,MRPC
                        help='Path to data_folder')
    parser.add_argument('--root', type=str,
                        default='/data/ljx/data/glue/data/k-shot',
                        help='')
    parser.add_argument('--origin_K', type=str,
                        default='16',
                        help='')
    parser.add_argument('--seeds', type=str,
                        default='13,21,42,87,100',
                        help='')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='')
    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help='')
    
    parser.add_argument('--decoding', type=str, default='sampling',
                        help='{greedy, sampling, beam_gen, beam_sample}')
    parser.add_argument('--beam_size', type=int, default=8,
                        help='Beam size for beam search decoding')
    parser.add_argument('--k', type=int, default=15,
                        help='k for top-k sampling (0 for deactivate)')
    parser.add_argument('--p', type=float, default=0.92,
                        help='p for necleus (top-p) sampling')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='temperature for sampling-based decoding')
    parser.add_argument('--num_generate', type=int, default=1,
                        help='How many sequences are generated')
    
    args = parser.parse_args()
    
    args.datasets = args.datasets.split(',')
    args.seeds = args.seeds.split(',')
    
    main(args)