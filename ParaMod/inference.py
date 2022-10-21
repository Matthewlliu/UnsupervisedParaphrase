import argparse
import csv
import json
import os
from datetime import datetime
import random
import logging

import numpy as np
import torch
from model.t5_paraphrase_model import t5_model

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M")

def filter_special_tokens(sent):
    sent = sent.strip().lower()
    if '<extra_id_0>' in sent:
        sent = sent.split('<extra_id_0>')[1]
    if '<extra_id_1>' in sent:
        sent = sent.split('<extra_id_1>')[0]
    while sent.endswith('</s>'):
        sent = sent[:-len('</s>')].strip()
    if sent[:1] == '"':
        sent = sent[1:]
    if sent[-2:] == '?.':
        sent = sent[:-1]
    sent = sent.replace('.', ' .')
    sent = sent.replace('?', ' ?')
    sent = ' '.join(sent.replace('<unk>', '').split(' '))
    return sent

def inference(args):
    t5 = t5_model(args)
    t5.build_model(args.checkpoint, with_tokenizer=False)
    
    #sentences = []
    with open(args.data_path, 'r') as f:
        sentences = f.readlines()
            
    if args.toy is True:
        sentences = sentences[:5]
    
    logging.info("START INFERENCE")
    print("START INFERENCE")
    seq_result = t5.generate_text(
        sentences,
        max_length=args.max_length,
        decoding=args.decoding
    )
    logging.info("DONE INFERENCE")
    print("DONE INFERENCE")
    logging.info("Save to {}".format(args.save))
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w') as f:
        for idx, generated in enumerate(seq_result):
            if isinstance(generated, list):
                for seq in generated:
                    f.write('{}\t{}\n'.format(idx, filter_special_tokens(seq)))
            else:
                f.write('{}\t{}\n'.format(idx, filter_special_tokens(generated)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./data/QQP_split/test_input_preprocessed.txt',
                        help='Dataset file to paraphrase')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to LOAD model checkpoint')
    parser.add_argument('--model', type=str, default='t5-base',
                        help='pretrained model name (to load tokenizer)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=str, default=None,
                        help='File name to save generated sentences')
    parser.add_argument('--log', type=str, default=None,
                        help='Log filename')

    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum number of tokens for each sequence')

    parser.add_argument('--decoding', type=str, default='sampling',
                        help='{greedy, sampling, beam_gen, beam_sample}')
    parser.add_argument('--beam_size', type=int, default=8,
                        help='Beam size for beam search decoding')
    parser.add_argument('--k', type=int, default=0,
                        help='k for top-k sampling (0 for deactivate)')
    parser.add_argument('--p', type=float, default=1.0,
                        help='p for necleus (top-p) sampling')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for sampling-based decoding')
    parser.add_argument('--num_generate', type=int, default=1,
                        help='How many sequences are generated')

    parser.add_argument('--tag', type=str, default='',
                        help='Add a suffix of checkpoints')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    args.decoding_name = args.decoding
    if args.decoding == 'beam':
        args.decoding_name += '-{}'.format(args.beam_size)
        raise NotImplementedError  # TODO
    elif args.decoding == 'sampling':
        args.decoding_name = 'top-{}'.format(args.k)
        args.decoding_name += '-p{}'.format(args.p).replace('.', '_')
        args.decoding_name += '-T{}'.format(args.temperature).replace('.', '_')
        
    filename = "inferenced_{}_seed{}_{}".format(
            args.decoding_name, args.seed, args.tag + '_' + start_datetime)

    if args.save is None:
        args.save = "./inference/{}.txt".format(filename)
    if args.log is None:
        args.log = args.save.replace('inference', 'logs')
        args.log = args.log.replace('.txt', '.log') #'./logs/{}.log'.format(filename)

    log_path = '/'.join(args.log.split('/')[:-1])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format=log_format, filename=args.log)
    logging.getLogger().setLevel(log_level)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info('Parsed args: ' + json.dumps(dict(args.__dict__), indent=2))

    inference(args)