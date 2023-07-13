from __future__ import print_function

import torch
from torch import nn, Tensor
import numpy as np

from model.base import BaseBuilder


class BMLBuilder(BaseBuilder):
    def __init__(self, opt):
        super().__init__(opt)
        if opt.backbone == 'Res12':
            from backbone.resnet12_bml import Res12Share, Res12Indep
            self.encoder = Res12Share()
            self.global_view = Res12Indep()
            self.local_view = Res12Indep()
            hdim = 640
        elif opt.backbone == 'Res18':
            from backbone.resnet18_bml import Res18Share, Res18Indep
            self.encoder = Res18Share()
            self.global_view = Res18Indep()
            self.local_view = Res18Indep()
            hdim = 512
        else:
            raise ValueError('')

        self.n_ways = opt.n_ways
        self.alpha_1 = opt.alpha_1
        self.alpha_2 = opt.alpha_2

        self.global_w = nn.Conv2d(in_channels=hdim, out_channels=self.opt.n_cls, kernel_size=1, stride=1)
        nn.init.xavier_uniform_(self.global_w.weight)

    def global_forward(self, x, split):
        global_feat = self.global_view(x) # -> [B, 640, 5, 5]
        if split in ["test", "eval"]:
            n_ways = self.opt.n_ways
            n_queries = self.opt.n_queries
            n_shots = self.opt.n_shots
            # global_logits = self._eval_matching(global_feat, n_ways, n_queries, n_shots) # 欧氏距离，且没有取根号。
            global_logits = 0

        else:
            global_logits = self.global_w(global_feat) # 1x1 cnn 相当于fc，在这里做(5x5)稠密分类。
            global_logits = global_logits.flatten(start_dim=2) # [180, 64, 25]
        return global_logits, global_feat.flatten(start_dim=2).mean(dim=-1)

    def local_forward(self, x, label, split, epoch, max_epoch):
        local_feat = self.local_view(x)
        if split in ["test", "eval"]:
            n_ways = self.opt.n_ways
            n_queries = self.opt.n_queries
            n_shots = self.opt.n_shots
            local_logits = 0 # 提取特征时要把这里注释掉
            # local_logits = self._eval_matching(local_feat, n_ways, n_queries, n_shots)
        else:
            n_ways = self.opt.n_train_ways
            n_queries = self.opt.n_train_queries
            n_shots = self.opt.n_train_shots
            # local_logits = self._matching(local_feat, label,
            #                               n_ways=n_ways,
            #                               n_queries=n_queries,
            #                               n_shots=n_shots,
            #                               epoch=epoch, max_epoch=max_epoch)
            local_logits = 0
            
        return local_logits, local_feat.flatten(start_dim=2).mean(dim=-1)

    def forward(self, x, label=None, split="test", epoch=0, max_epoch=200):
        share_feat = self.encoder(x) # [B, 3, 84, 84] -> [B, 320, 10, 10]
        global_logits, global_feat = self.global_forward(share_feat, split) 
        local_logits, local_feat = self.local_forward(share_feat, label, split, epoch, max_epoch)
        return global_logits, local_logits, global_feat, local_feat

    def _matching(self, feat, label, n_ways, n_queries, n_shots, epoch, max_epoch):
        bs, emb, w, h = feat.shape
        support_idx, query_idx = self.split_instances(n_ways, n_queries, n_shots)

        logits = torch.zeros((n_queries * n_ways, n_ways, w * h)).cuda()
        emb_dim = feat.size(1)
        support = feat[support_idx.flatten()].view(*(support_idx.shape + (emb, w * h)))
        query = feat[query_idx.flatten()].view(*(query_idx.shape + (emb, w * h)))
        proto = support.mean(dim=1).mean(dim=-1) # 这里求了两次均值
        for i in range(query.size(-1)): # 计算query feature map上每个位点之于 prototypes 的距离。
            cur_logits = self.compute_logits(proto, query[..., i], emb_dim, query_idx)
            constraint = self._elastic_constraint(n_ways, cur_logits, label, epoch, max_epoch)
            cur_logits = - (cur_logits + constraint)
            logits[..., i] = cur_logits
        return logits

    def _eval_matching(self, instance_embs, n_ways, n_queries, n_shots):
        instance_embs = torch.relu(instance_embs).pow(0.5) # ????
        instance_embs = instance_embs.flatten(start_dim=1)
        # instance_embs = instance_embs.flatten(start_dim=2).mean(dim=-1)
        support_idx, query_idx = self.split_instances(n_ways, n_queries, n_shots)
        emb = instance_embs.size(-1)
        support = instance_embs[support_idx.flatten()].view(
            *(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
        proto = support.mean(dim=1)
        logits = - self.compute_logits(proto, query, emb, query_idx)
        return logits

    def _elastic_constraint(self, n_ways, logits, label, epoch, max_epoch):
        ep_weight = epoch / max_epoch
        constraint = torch.zeros_like(logits).cuda()
        for idx in range(logits.shape[0]):
            cur_dis = logits[idx].detach().cpu()
            cur_label = label[idx].detach().cpu()
            pos_dis = cur_dis[cur_label] 
            mask = (torch.arange(0, n_ways) == cur_label).int()
            neg_hard_dis, _ = torch.topk(cur_dis * (1 - mask), k=2, largest=False)  # hard negative
            assert neg_hard_dis[0] == 0 and _[0] == cur_label
            prob = torch.softmax(-cur_dis, dim=0) # 这里没有加 温度，是否可以加温度给 hard negative更大的惩罚。
            prob_pos, prob_neg = prob[_[0]], prob[_[1]]

            if self.alpha_1 == 0:  # baseline
                constraint[idx] = mask * self.alpha_1
                assert not constraint[idx].bool().all()
            else:
                if prob_neg > 0.5 or prob_pos > 0.5:  # discard easyest/hardest task # 这个策略很重要
                    constraint[idx] = 0
                else:
                    # diff: pos_dis - neg_hard_dis[1] represents difficulty
                    # bigger diff get bigger beta (weights), focus on harder task
                    beta = torch.sigmoid(self.alpha_2 * (pos_dis - neg_hard_dis[1]))
                    constraint[idx] = mask * (ep_weight + 1e-8) * beta * self.alpha_1
        return constraint
