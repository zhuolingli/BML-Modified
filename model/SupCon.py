from __future__ import print_function

import torch
from torch import nn, Tensor
import numpy as np

from model.base import BaseBuilder
import torch.nn.functional as F




class SeflAttention(nn.Module):
    pass


class BaseSupConBuilder(BaseBuilder):
    def __init__(self, opt):
        super().__init__(opt)
        if opt.backbone == 'Res12':
            from backbone.resnet12 import resnet12
            self.encoder = resnet12()
            hdim = 640
            
        elif opt.backbone == 'Res18':
            from backbone.resnet18 import resnet18
            self.encoder = resnet18()
            hdim = 512
        else:
            raise ValueError('')

        # self.self_att = SeflAttention()
        # projection head need to init?
        self.prohead = nn.Sequential(
                nn.Conv2d(hdim, hdim, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hdim, opt.latent_dim, kernel_size=1, stride=1),)
        for layer in self.prohead:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, x, split="test"):
        instance_embs, instance_embs_spatial = self.encoder(x)
        if split == "train":
            if self.opt.spatial:
                latent_feature = self.prohead(instance_embs_spatial) # project to latent space。 [B, 128, 5, 5]
                latent_feature = F.normalize(latent_feature, dim=1) # [B, 128, 5, 5]
                
            else:
                latent_feature = self.prohead(instance_embs) # project to latent space。 [B, 128, 1]
                latent_feature = F.normalize(latent_feature, dim=1)
            return latent_feature
        else:
            support_idx, query_idx = self.split_instances(self.opt.n_ways, self.opt.n_queries, self.opt.n_shots)
            emb = instance_embs.size(-1)
            support = instance_embs[support_idx.flatten()].view(
                *(support_idx.shape + (-1,)))
            query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
            proto = support.mean(dim=1)
            logits = - self.compute_logits(proto, query, emb, query_idx)
        return logits
        
    