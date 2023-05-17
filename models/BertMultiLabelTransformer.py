from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


class BertMultiLabelTransformer(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super(BertMultiLabelTransformer, self).__init__()

        # self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_encoder = BertModel.from_pretrained(bert_model_name)

        self.classifier = nn.Sequential(
            nn.Linear(self.bert_encoder.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_labels),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        bert_output = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        logits = self.classifier(pooled_output)

        return logits
