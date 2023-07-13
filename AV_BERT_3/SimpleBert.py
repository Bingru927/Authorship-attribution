import torch
import torch.nn as nn
from transformers import BertModel

class SimpleBert(nn.Module):
    def __init__(self, num_classes):
        super(SimpleBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.linear(pooled_output)
        return logits


if __name__ == '__main__':
    model = SimpleBert(2)
    x = torch.randint(0, 100, [32, 100])
    m = torch.randint(0, 1, [32, 100])
    y = model(x,m)
    print(y.shape)