
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import  AutoModel, AutoConfig, AutoModelForSequenceClassification
from torch.autograd import Function
from torch.nn import Parameter
import math

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class FineTuning(nn.Module):
    def __init__(self, args):
        super(FineTuning, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.to('cuda')

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs['logits'], None, None

class MyIGD(nn.Module):

    def __init__(self, args,s=1.0):
        super(MyIGD, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder = AutoModel.from_pretrained(args.pretrained_bert_name, config=config)

        self.s = s
        self.m = args.m
        self.weight = Parameter(torch.Tensor(args.lebel_dim, config.hidden_size*2 ))
        nn.init.xavier_uniform_(self.weight)
        self.norm = nn.BatchNorm1d(config.hidden_size * 2)

        if not args.inactivate_data_loss:

            self.discriminator = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size* 2),
                nn.BatchNorm1d(config.hidden_size* 2),
                nn.ReLU(True),
                nn.Linear(config.hidden_size* 2, 2),
            )

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1, w2).clamp(min=eps)


    def forward(self, inputs, label=None, alpha=None):
        if len(inputs) > 5:
            input_ids, token_type_ids, attention_mask, input_ids_, token_type_ids_, attention_mask_, label = inputs
            outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
                'pooler_output']
            outputs_ = self.encoder(input_ids_, token_type_ids=token_type_ids_, attention_mask=attention_mask_)[
                'pooler_output']
            outputs = torch.cat([outputs, outputs_], dim=-1)
        else:
            input_ids, token_type_ids, attention_mask = inputs[:3]
            outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
                'pooler_output']
            outputs = outputs.repeat(1, 2)

        logits_revers = None
        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(torch.clone(outputs), alpha)
            logits_revers = self.discriminator(reverse_feature)

        self.norm(outputs)
        cosine = self.cosine_sim(outputs, self.weight)
        if label is not None:
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)
            logits = self.s * (cosine - one_hot * self.m)
        else:
            logits = cosine

        return logits, outputs,logits_revers



