# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import sys
sys.path.append('..')

from transformers.file_utils import cached_path, PYTORCH_PRETRAINED_BERT_CACHE
from transformers.modeling_roberta import RobertaModel, RobertaLMHead
from transformers.modeling_bert import BertPreTrainedModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
    
class RobertaForCloth(BertPreTrainedModel):

    def __init__(self, config):
        super(RobertaForCloth, self).__init__(config)
        
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.vocab_size = self.roberta.embeddings.word_embeddings.weight.size(0)
    
    def get_output_embeddings(self):
        return self.lm_head.decoder
    
    def accuracy(self, out, tgt):
        out = torch.argmax(out, -1)
        return (out == tgt).float()
        
    def forward(self, inp, tgt):
        '''
        input: article -> bsz X alen, 
        option -> bsz X opnum X 4 X olen
        output: bsz X opnum 
        '''
        articles, articles_mask, ops, ops_mask, question_pos, mask, high_mask = inp 
        
        bsz = ops.size(0)
        opnum = ops.size(1)
        outputs = self.roberta(articles, attention_mask = articles_mask)
        out, _ = outputs[0]
        question_pos = question_pos.unsqueeze(-1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)
        out = self.lm_head(out)
        #convert ops to one hot
        out = out.view(bsz, opnum, 1, self.vocab_size)
        out = out.expand(bsz, opnum, 4, self.vocab_size)
        out = torch.gather(out, 3, ops)
        #mask average pooling
        out = out * ops_mask
        out = out.sum(-1)
        out = out/(ops_mask.sum(-1))
        
        out = out.view(-1, 4)
        tgt = tgt.view(-1,)
        loss = self.loss(out, tgt)
        acc = self.accuracy(out, tgt)
        loss = loss.view(bsz, opnum)
        acc = acc.view(bsz, opnum)
        loss = loss * mask
        acc = acc * mask
        acc = acc.sum(-1)
        acc_high = (acc * high_mask).sum()
        acc = acc.sum()
        acc_middle = acc - acc_high

        loss = loss.sum()/(mask.sum())
        loss = 0
        return loss, acc, acc_high, acc_middle
                           
    def init_zero_weight(self, shape):
        weight = next(self.parameters())
        return weight.new_zeros(shape)    
    
if __name__ == '__main__':
    bsz = 32
    max_length = 50
    max_olen = 3
    articles = torch.zeros(bsz, max_length).long()
    articles_mask = torch.ones(articles.size())
    ops = torch.zeros(bsz, 4, max_olen).long()
    ops_mask = torch.ones(ops.size())
    question_id = torch.arange(bsz).long()
    question_pos = torch.arange(bsz).long()
    ans = torch.zeros(bsz).long()
    inp = [articles, articles_mask, ops, ops_mask, question_id, question_pos]
    tgt = ans
    model = RobertaForCloth.from_pretrained('/chpc/home/stu-ysfang-a/RoBERTa-data/roberta-large',
          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
    loss, acc = model(inp, tgt)