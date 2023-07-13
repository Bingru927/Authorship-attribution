import torch
from configs import *
from torch import nn
from transformers import BertModel


class BertCls(nn.Module):
    """
    Bert Clf model
    """
    def __init__(self, num_classes):
        super(BertCls, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(bert_model_output_dim, num_hiddens)
        self.decoder = nn.Linear(num_hiddens * 2, num_classes)

    def forward(self, input_ids, attention_mask, tags=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_output = bert_output.last_hidden_state
        bert_output = self.dropout(bert_output)
        bert_output = bert_output.permute(1, 0, 2)

        outputs, _ = self.encoder(bert_output)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


if __name__ == '__main__':
    model = BertCls(2)
    x = torch.randint(0, 100, [32, 100])
    m = torch.randint(0, 1, [32, 100])
    y = model(x, m)
    print(y.shape)