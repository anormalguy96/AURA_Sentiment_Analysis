# src/model.py

import torch
import torch.nn as nn
from transformers import BertModel
from . import config

class AURA(nn.Module):
    def __init__(self, n_classes):
        super(AURA, self).__init__()
        self.bert = BertModel.from_pretrained(config.MODEL_NAME)
        
        self.drop = nn.Dropout(p=0.3)
        
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)