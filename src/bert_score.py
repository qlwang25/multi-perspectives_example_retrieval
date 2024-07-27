# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
sys.path.append('../')
sys.path.append("../../")
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformer_utils.models.bert.tokenization_bert import BertTokenizer
from utils import (convert_examples_to_features, EntityProcessor)


logger = logging.getLogger(__name__)
torch.set_num_threads(12)


class BertForSequenceLabeling(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceLabeling, self).__init__(config)
        config.clue_num = 0
        self.config = config
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pool_out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False,)
        pool_out = self.dropout(pool_out)
        return  pool_out


def evaluate(args, model, tokenizer, eval_dataset, out_dir):
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.batch_size)
    model.eval()
    out_reps = []
    for batch in eval_dataloader:
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            out = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            out_reps.append(out)
    all_tensor = torch.cat(out_reps, dim=0)
    print(all_tensor.shape)
    return all_tensor


def load_and_cache_examples(args, tokenizer, data_dir, dataname="train"):
    logger.info("Creating features from dataset file at %s", data_dir)
    if dataname == "train":
        examples = args.processor.get_examples(data_dir, "train")
    elif dataname == "test":
        examples =args.processor.get_examples(data_dir, "test")

    features = convert_examples_to_features(examples, args.label2id, args.max_seq_length, tokenizer, 
        cls_token = tokenizer.cls_token, 
        sep_token = tokenizer.sep_token,
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        sequence_a_segment_id = 0, 
        pad_token_label_id = args.pad_token_label_id,
        )

    all_inputs = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segments = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_inputs, all_masks, all_segments)
    return dataset


def calculate_similarity(tensor_A, tensor_B):
    norm_A = torch.nn.functional.normalize(tensor_A, p=2, dim=1)
    norm_B = torch.nn.functional.normalize(tensor_B, p=2, dim=1)
    similarity_matrix = torch.mm(norm_A, norm_B.t())
    return similarity_matrix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='../data/coling/', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--pad_token_label_id", default=-1, type=int, help="id of pad token .")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    file2model = {
            "lap14": "/data/coling/lap14",
        }
    
    args.processor = EntityProcessor()
    args.label2id = args.processor.get_labels()
    for file, model_path in file2model.items():
        tokenizer = BertTokenizer(os.path.join(model_path, "vocab.txt"), do_lower_case=True)
        model = BertForSequenceLabeling.from_pretrained(model_path)
        model.to(args.device)

        train_datasets = load_and_cache_examples(args, tokenizer, data_dir=args.data_dir+file, dataname="train")
        eval_datasets = load_and_cache_examples(args, tokenizer, data_dir=args.data_dir+file, dataname="test")

        trains = evaluate(args, model, tokenizer, train_datasets, out_dir=os.path.join(args.data_dir, file, "trains.pt"))
        tests = evaluate(args, model, tokenizer, eval_datasets, out_dir=os.path.join(args.data_dir, file, "evals.pt"))
        trains = trains[:100, :]

        similarity_matrix = calculate_similarity(tests, trains)
        result_json = {}
        for i, similarity_scores in enumerate(similarity_matrix):
            sorted_indices = torch.argsort(similarity_scores, descending=True)
            sorted_indices_list = sorted_indices.tolist()
            result_json[f"{i}"] = [f"{idx}" for idx in sorted_indices_list]

        with open(os.path.join(args.data_dir, file, file+"_similarity_results.json"), "w") as f:
            json.dump(result_json, f)


if __name__ == "__main__":
    main()
