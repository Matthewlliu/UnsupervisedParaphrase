import argparse
import json
from datetime import datetime
import random
import logging
import os

import numpy as np
import torch
from transformers import Seq2SeqTrainingArguments as HfTrainingArguments
from transformers import Seq2SeqTrainer, TrainingArguments, Trainer, HfArgumentParser
from transformers import set_seed

from model.t5_paraphrase_model import t5_model
from model.bart_paraphrase_model import bart_model
from model.gpt2_paraphrase_model import gpt2_model
from para_loader import Para_dataset

import deepspeed
import dataclasses
from dataclasses import dataclass, field, fields

start_datetime = datetime.now().strftime("%Y-%m-%d")

def main(model_args, data_args, train_args):
    if "t5" in model_args.model:
        lm = t5_model(model_args)
    elif "bart" in model_args.model:
        lm = bart_model(model_args)
    elif "gpt2" in model_args.model:
        lm = gpt2_model(model_args)
    else:
        raise ValueError("Other LMs are not implemented")
    lm.build_model(checkpoint_dir=model_args.checkpoint)
    
    file_list = [os.path.join(data_args.train_dir, 'train_finetune.txt')] if data_args.finetune else os.listdir(data_args.train_dir)
    print(file_list)
    train_dataset = Para_dataset(lm.tokenizer, file_list, data_args.train_dir,
                                 device=model_args.device, finetune=data_args.finetune, 
                                 sample_num=data_args.sample_num)
    logging.info("Start training")
    last_step = 0
    
    training_args = TrainingArguments(
        output_dir=train_args.output_dir,
        num_train_epochs=train_args.num_train_epochs,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        learning_rate=train_args.learning_rate,
        warmup_steps=300,
        weight_decay=0.01,
        evaluation_strategy='no',
        save_steps=train_args.save_steps,
        save_total_limit=train_args.save_total_limit,
        #eval_steps=args.save_steps,
        seed=train_args.seed,
        logging_dir='./logs',
        do_train=True,
        fp16 = train_args.fp16,
        deepspeed = train_args.deepspeed,
    )

    trainer = Seq2SeqTrainer(
        model=lm.model,
        args=train_args,
        train_dataset=train_dataset,
        #eval_dataset=dev_dataset,
        #tb_writer=t5.writer,
        #prediction_loss_only=True,
    )
    trainer.train()
    trainer.save_model()

    #trainer.evaluate()

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='./data/run_311')
    parser.add_argument('--dev_data_path', type=str,
                        default='./data/run_311')
    parser.add_argument('--checkpoint', type=str,
                        default=None)
    parser.add_argument('--save_dir', type=str, 
                        default='/data/ljx/result/para_model/{}/checkpoints/')
    parser.add_argument('--summary_dir', type=str, default=None,
                        help='Path to save tensorboard summary')
    parser.add_argument('--log', type=str,
                        default='./logs/train_{datetime}.log')
    parser.add_argument('--device', type=str, 
                        default='cuda')
    parser.add_argument('--model', type=str,
                        default='t5-large')
    parser.add_argument('--max_length', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=4)
    parser.add_argument('--eval_batch_size', type=int,
                        default=4)
    parser.add_argument('--gradient_accumulation', type=int,
                        default=1)
    parser.add_argument('--learning_rate', type=float,
                        default=5)
    parser.add_argument('--num_epochs', type=int,
                        default=10)
    parser.add_argument('--warmup_ratio', type=float,
                        default=0.002)
    parser.add_argument('--save_steps', type=int, 
                        default=500)
    parser.add_argument('--save_total_limit', type=int, 
                        default=5)
    parser.add_argument('--tag', type=str,
                       help='add a unique suffix')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--deepspeed',type=str, default=None)
    args = parser.parse_args()
    '''

@dataclass
class ModelArguments:
    model: str = field(
        default="t5-base",
        metadata={"help": "name or path to the model"}
    )
    checkpoint: str = field(
        default=None,
        metadata={"help": "path to a model checkpoint"}
    )
    tag: str = field(
        default="",
        metadata={"help": "Tag for Naming"}
    )
    max_length: int = field(
        default=512, 
        metadata={"help": "max length of input sequence after tokenization"}
    )
    device: str = field(
        default='cuda',
        metadata={"help": "available device"}
    )
    summary_dir: str = field(
        default=None,
        metadata={"help": "summary writing directory"}
    )

@dataclass
class DataArguments:
    train_dir: str = field(
        default='./data/run_311',
        metadata={"help": "training set directory"}
    )
    eval_dir: str = field(
        default=None,
        metadata={"help": "validation set directory"}
    )
#    summary_dir: str = field(
#        default=None,
#        metadata={"help": "summary directory"}
#    )
    log: str = field(
        default='./logs/train_{datetime}.log',
        metadata={"help": "log directory"}
    )
    #debug: bool = field(
    #    default=False,
    #    metadata={"help": "log debugging config"}
    #)
    toy: bool = field(
        default=False,
        metadata={"help": "whether to play with small subset of the data for debugging"}
    )
    finetune: bool = field(
        default=False,
        metadata={"help": "mark for finetuning"}
    )
    sample_num: int = field(
        default=500, 
        metadata={"help": "sample number for few-shot finetuning"}
    )


@dataclass
class TrainingArguments(HfTrainingArguments):
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )
    # Turn off train/test
    seed: int = field(
        default=1234,
        metadata={"help": "set seed for reproducibility"}
    )
        
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    
    # process output_dir
    save_dir = model_args.model
    save_dir += '_toy' if data_args.toy else ''
    save_dir += '_ft' if data_args.finetune else ''
    save_dir += '-{}'.format(model_args.tag) if model_args.tag else ''
    save_dir += '_continued' if model_args.checkpoint is not None else ''
    save_dir += '_{}'.format(start_datetime)
    print("Save folder name: {}".format( save_dir ))
    train_args.output_dir = train_args.output_dir.format( save_dir )
        
    if model_args.summary_dir is None:
        model_args.summary_dir = os.path.join(train_args.output_dir, 'runs')
    
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if train_args.debug else logging.INFO
    log_file = data_args.log
    if model_args.tag:
        log_file = log_file.replace('{datetime}', model_args.tag + '_{datetime}')
    logging.basicConfig(level=log_level, format=log_format,
                        filename=log_file.format(datetime=start_datetime))
    logging.getLogger().setLevel(log_level)

    # Reproducibility
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    set_seed(train_args.seed)

    if data_args.toy:
        data_args.train_dir = data_args.eval_dir
    logging.info('Parsed args: ' + json.dumps(dict(model_args.__dict__), indent=2))
    logging.info(json.dumps(dict(data_args.__dict__), indent=2))
    #logging.info(json.dumps(dict(train_args.__dict__), indent=2))

    main(model_args, data_args, train_args)