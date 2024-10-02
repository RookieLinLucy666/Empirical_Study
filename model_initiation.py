# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:09:20 2020

@author: user
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataclasses import dataclass
import math
from torchvision import models
import torchvision.models as models


def model_init(data_name):
    if(data_name == 'mnist'):
        model = Net_mnist()
    elif(data_name == 'fmnist'):
        model = Net_mnist()
    elif(data_name == 'cifar10'):
        model = Net_cifar10()
    elif(data_name == 'purchase'):
        model = Net_purchase()
    elif(data_name == 'adult'):
        model = Net_adult()
    elif(data_name == 'shakespeare'):
        model = BabyGPTmodel(config)
    elif(data_name == 'cifar100'):
        # 加载预定义的 MobileNetV2 模型
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 100)
    return model

# class Net_mnist(nn.Module):
#     def __init__(self):
#         super(Net_mnist, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net_purchase(nn.Module):
    def __init__(self):
        super(Net_purchase, self).__init__()
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50,2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x   
    

class Net_adult(nn.Module):
    def __init__(self):
        super(Net_adult, self).__init__()
        self.fc1 = nn.Linear(108, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10,2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

# 定义全局模型架构
class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 输出尺寸: (batch_size, 32, 16, 16)
        x = self.pool(torch.relu(self.conv2(x)))  # 输出尺寸: (batch_size, 64, 8, 8)
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class Net_cifar10(nn.Module):
#     def __init__(self):
#         super(Net_cifar10, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class All_CNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(All_CNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )
      
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output  
    
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model) 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x  
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)



@dataclass
class GPTConfig:
    # these are default GPT-2 hyperparameters
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias :bool = False

config = GPTConfig(
    block_size = 4,
    vocab_size = 109,
    n_head = 4,
    n_layer = 4,
    n_embd = 16)



class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        assert config.n_embd % config.n_head == 0

        self.atten = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B,T,C = x.size()
        q, k ,v  = self.atten(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


        # manual implementation of attention
        # from karpathy
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.projection(y)
        return y
dropout=0.2
class FeedForward(nn.Module):
    def __init__(self,config):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                                 nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                                 nn.GELU(),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

### A simple Transformer Block
class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer, self).__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_1 = nn.LayerNorm(config.n_embd)
        self.layer_norm_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):

        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class BabyGPTmodel(nn.Module):

    def __init__(self, config):
        super(BabyGPTmodel, self).__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Transformer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps = 1e-12) # final layer norm
        self.lnum_heads = nn.Linear(config.n_embd, config.vocab_size)

        ## init all weights
        ## from karpathy
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('projection.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        # print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters()),))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        tok_emb = self.token(idx)
        position_ids = torch.arange(0, T, dtype = torch.long, device = device).unsqueeze(0)
        pos_emb =  self.positional_embeddings(position_ids)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lnum_heads(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    ## from karpathy's youtube videos.
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
