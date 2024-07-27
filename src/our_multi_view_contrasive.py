# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import random
import sys
sys.path.append('../')
sys.path.append("../../")
import json
import nltk

import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformer_utils.models.bert.tokenization_bert import BertTokenizer
from optimization import AdamW, WarmupLinearSchedule, Warmup
from utils import EntityProcessor, InputExample


logger = logging.getLogger(__name__)
torch.set_num_threads(12)


class BertForSequence(BertPreTrainedModel):
    def __init__(self, config, tokenizer, max_seq_length):
        super(BertForSequence, self).__init__(config)
        config.clue_num = 0
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ex_index = 0
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def convert_text_to_feature(self, text):
        cls_token = '[CLS]'
        sep_token = '[SEP]'
        pad_token_id = 0

        tokens = []
        for tok in text.split():
            tokens.extend(self.tokenizer.wordpiece_tokenizer.tokenize(tok))
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:(self.max_seq_length - 2)]

        tokens = [cls_token] + tokens + [sep_token]        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0] * padding_length)
        assert len(input_ids) == len(input_mask) == self.max_seq_length
        
        if self.ex_index < 1:
            logger.info("tokens: %s" % text)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            self.ex_index += 1
        return input_ids, input_mask

    def get_path_representations(self, indexs, examples, flag):
        all_input_ids, all_input_masks = [], []
        for index in indexs:
            if flag == 'text':
                text = examples[index].text_a

            if flag == 'label':
                sents = [f'the aspect " {aspect} " is {sentiment}' for aspect, sentiment in examples[index].label_a]
                text = 'In this sentence , ' + ' , '.join(sents) + ' .'
            
            if flag == 'pos':
                words = nltk.word_tokenize(examples[index].text_a)
                pos_tags = nltk.pos_tag(words)
                text = ' '.join([p for w, p in pos_tags])

            text = text.lower()
            input_id, input_mask = self.convert_text_to_feature(text)
            input_id = torch.tensor(input_id).unsqueeze(0).cuda()
            input_mask = torch.tensor(input_mask).unsqueeze(0).cuda()

            all_input_ids.append(input_id)
            all_input_masks.append(input_mask)

        input_ids = torch.cat(all_input_ids, dim=0)
        input_masks = torch.cat(all_input_masks, dim=0)
        _, pool_output = self.bert(input_ids=input_ids, attention_mask=input_masks, output_all_encoded_layers=False)
        hiddens = self.dropout(pool_output)
        return hiddens

    def infonce_loss(self, T1, T2, temperature=0.5):
        assert T1.size() == T2.size()
        batch_size = T1.size(0)
        cosine_similarity = F.cosine_similarity(T1.unsqueeze(1), T2.unsqueeze(0), dim=2)
        labels = torch.arange(batch_size).to(T1.device)
        cosine_similarity /= temperature
        loss = F.cross_entropy(cosine_similarity, labels)
        return loss


def train(args, model, samples):
    indexs = torch.arange(len(samples), dtype=torch.long)
    INDEX = TensorDataset(indexs)
    index_dataloader = DataLoader(INDEX, sampler=RandomSampler(INDEX), batch_size=args.batch_size)
    num_optimization_steps = len(index_dataloader) * args.num_train_epochs

    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(samples))
    logger.info("Total optimization steps = %d", num_optimization_steps)

    model.zero_grad()
    model.train()
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        for _, batch in enumerate(index_dataloader):
            one_embeds = model.get_path_representations(batch[0].tolist(), samples, flag='text')
            two_embeds = model.get_path_representations(batch[0].tolist(), samples, flag='text')
            loss = model.infonce_loss(one_embeds, two_embeds)

            label_embeds = model.get_path_representations(batch[0].tolist(), samples, flag='label')
            loss2 = model.infonce_loss(one_embeds, label_embeds)

            pos_embeds = model.get_path_representations(batch[0].tolist(), samples, flag='pos')
            loss3 = model.infonce_loss(one_embeds, pos_embeds)

            (loss + 0.1*loss2 + 0.1*loss3).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}, Loss2:{:.5f}, Loss3:{:.5f}".format(epoch, global_step, num_optimization_steps, loss.item(), loss2.item(), loss3.item()))
            global_step += 1


def evaluate(args, model, samples, out_dir, flag='text'):
    indexs = torch.arange(len(samples), dtype=torch.long)
    INDEX = TensorDataset(indexs)
    eval_dataloader = DataLoader(INDEX, sampler=SequentialSampler(INDEX), batch_size=args.batch_size)
    
    model.eval()
    out_reps = []
    for batch in eval_dataloader:
        with torch.no_grad():
            out = model.get_path_representations(batch[0].tolist(), samples, flag=flag)
            out_reps.append(out)

    all_tensor = torch.cat(out_reps, dim=0)
    print(all_tensor.shape)
    torch.save(all_tensor, out_dir)


def load_and_cache_examples(args, file, dataname="train", K=-1):
    logger.info("Creating features from dataset file at %s", args.data_dir)
    with open(os.path.join(args.data_dir, file, dataname+".json"), 'r') as json_file:
        datas = json.load(json_file)

    examples = []
    N = K if K != -1 else len(datas) 
    for i in range(N):
        guid = "%s-%s" % (dataname, i)
        toks = datas[str(i)]['tokens']
        label = datas[str(i)]['tag']
        if i < 2:
            print(toks, label)
        examples.append(InputExample(guid=guid, text_a=toks, label_a=label))
    logger.info("load file :{}, size :{}".format(os.path.join(args.data_dir, file, dataname+".json"), len(examples)))
    return examples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='../data/coling/', type=str)

    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str,
                        help="Can be `'WarmupLinearSchedule'`, `'warmup_constant'`, `'warmup_cosine'` , `None`, 'warmup_cosine_warmRestart' or a `warmup_cosine_hardRestart`")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--pad_token_label_id", default=-1, type=int, help="id of pad token .")
    parser.add_argument("--logging_global_step", default=200, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    file2model = {
            "lap14":"/pre-trained_model/bert_uncased_L-12_H-768_A-12",
        }

    args.processor = EntityProcessor()
    for file, model_path in file2model.items():
        bert_tokenizer = BertTokenizer(os.path.join(model_path, "vocab.txt"), do_lower_case=True)
        bert_model = BertForSequence.from_pretrained(model_path, tokenizer=bert_tokenizer, max_seq_length=args.max_seq_length)
        bert_model.cuda()

        examples = load_and_cache_examples(args, file, dataname="train", K=100)
        train(args, bert_model, examples)

        examples = load_and_cache_examples(args, file, dataname="train")
        test_examples = load_and_cache_examples(args, file, dataname="test")
        evaluate(args, bert_model, test_examples, out_dir=os.path.join(args.data_dir, file, "our_evals_100.pt"))


if __name__ == "__main__":
    main()


