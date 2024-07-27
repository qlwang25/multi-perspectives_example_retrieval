# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from __future__ import absolute_import, division, print_function

import logging
import os
from collections import defaultdict
import sys
sys.path.append("../")
sys.path.append("../../")
import json
from random import shuffle
import random
import re


logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, label_a):
        self.guid = guid
        self.text_a = text_a
        self.label_a = label_a


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, tokens_len=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.tokens_len = tokens_len


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    @classmethod
    def _read_txt(cls, input_path):
        with open(input_path, 'r') as json_file:
            datas = json.load(json_file)
        lines = []
        for k in range(len(datas)):
            lines.append((datas[str(k)]['tokens'], datas[str(k)]['BIOT']))
        return lines


class EntityProcessor(DataProcessor):
    def get_examples(self, data_dir, data_name):
        return self._create_examples(self._read_txt(os.path.join(data_dir, data_name+".json")), data_name)
    def get_labels(self):
        return {"B-POS":0, "I-POS":1, "B-NEU":2, "I-NEU":3, "B-NEG":4, "I-NEG":5, "O":6}
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            sent, tagt = line
            toks = sent.lower().split()
            assert len(toks) == len(tagt)
            examples.append(InputExample(guid=guid, text_a=toks, label_a=tagt))
        return examples

def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer,
    cls_token='[CLS]', 
    sep_token='[SEP]', 
    pad_token_id=0,
    sequence_a_segment_id=0, 
    sequence_b_segment_id=1,
    pad_token_label_id=-1,
    ):

    def _reseg_token_label(tokens, labels):
        assert len(tokens) == len(labels)
        ret_tokens, ret_labels = [], []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            ret_labels.append(label)

            if len(sub_token) == 1:
                continue
            
            if label.startswith("B") or label.startswith("I"):
                sub_label = "I-" + label[2:]
                ret_labels.extend([sub_label] * (len(sub_token) - 1))
            elif label.startswith("O"):
                sub_label = label
                ret_labels.extend([sub_label] * (len(sub_token) - 1))
            else:
                raise ValueError
        
        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def inputIdMaskSegment(tmp_tokens, tmp_labels, tmp_segment_id):
        tokens, labels = _reseg_token_label(tmp_tokens, tmp_labels)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            labels = labels[:(max_seq_length - 2)]

        label_ids = [label2id[label] for label in labels]
        pad_label_id = pad_token_label_id

        tokens = [cls_token] + tokens + [sep_token]
        segment_ids = [tmp_segment_id] * len(tokens)
        label_ids = [pad_label_id] + label_ids + [pad_label_id]

        tokens_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([tmp_segment_id] * padding_length)
        label_ids += ([pad_label_id] * padding_length)
        assert len(input_ids) == len(input_mask) == len(segment_ids) == len(label_ids) == max_seq_length
        return tokens, input_ids, input_mask, segment_ids, label_ids, tokens_len

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = example.text_a
        labels_a = example.label_a

        tokens, input_ids, input_mask, segment_ids, label_ids, tokens_len = inputIdMaskSegment(tokens_a, labels_a, sequence_a_segment_id)
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, tokens_len=tokens_len,))

    return features


