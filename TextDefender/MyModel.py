import torch
import torch.nn as nn
from transformers import  AutoModel, AutoConfig
from torch.nn import Parameter

class MyIGD(nn.Module):
    def __init__(self, args,s=1.0, m=0.3):
        super(MyIGD, self).__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = args.lebel_dim
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        self.args= args
        self.s = s
        self.m = m
        self.inactivate_lm_loss =args.inactivate_lm_loss
        self.norm = nn.BatchNorm1d(config.hidden_size * 2)
        self.weight = Parameter(torch.Tensor(args.lebel_dim, config.hidden_size*2 ))

        nn.init.xavier_uniform_(self.weight)

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1, w2).clamp(min=eps)


    def forward(self, inputs):
        if self.args.model_type =='bert':
            outputs = self.encoder(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask'])['pooler_output']
        elif self.args.model_type == 'roberta':
            outputs = self.encoder(input_ids=inputs['input_ids'],  attention_mask=inputs['attention_mask'])['pooler_output']

        outputs = outputs.repeat(1, 2)
        outputs=self.norm(outputs)
        logits = self.cosine_sim(outputs, self.weight)
        return logits,None


