#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import *
import random
import pickle
import math


def Identity(x):
    return x


# use box offset as covariance matrix.

# not use in the code
class SetIntersection(nn.Module):
    def __init__(self, mode_dims, expand_dims, agg_func=torch.min):
        super(SetIntersection, self).__init__()
        self.agg_func = agg_func
        self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat", self.pre_mats)
        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat", self.post_mats)
        self.pre_mats_im = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.pre_mats_im)
        self.register_parameter("premat_im", self.pre_mats_im)
        self.post_mats_im = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats_im)
        self.register_parameter("postmat_im", self.post_mats_im)

    def forward(self, embeds1, embeds2, embeds3=[], name='real'):
        if name == 'real':
            temp1 = F.relu(embeds1.mm(self.pre_mats))
            temp2 = F.relu(embeds2.mm(self.pre_mats))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats)

        elif name == 'img':
            temp1 = F.relu(embeds1.mm(self.pre_mats_im))
            temp2 = F.relu(embeds2.mm(self.pre_mats_im))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats_im))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats_im)
        return combined


# not used in this paper
class CenterSet(nn.Module):
    # torch min will just return a single element in the tensor
    def __init__(self, mode_dims, expand_dims, center_use_offset, agg_func=torch.min, bn='no', nat=1,
                 name='Real_center'):
        super(CenterSet, self).__init__()
        assert nat == 1, 'vanilla method only support 1 nat now'
        self.center_use_offset = center_use_offset
        self.agg_func = agg_func
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))

        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat_%s" % name, self.pre_mats)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn2 = nn.BatchNorm1d(mode_dims)
            self.bn3 = nn.BatchNorm1d(mode_dims)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        # every time the initial is different
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s" % name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        if self.center_use_offset:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1
            temp2 = embeds2
            if len(embeds3) > 0:
                temp3 = embeds3

        if self.bn == 'no':
            temp1 = F.relu(temp1.mm(self.pre_mats))
            temp2 = F.relu(temp2.mm(self.pre_mats))
        elif self.bn == 'before':
            temp1 = F.relu(self.bn1(temp1.mm(self.pre_mats)))
            temp2 = F.relu(self.bn2(temp2.mm(self.pre_mats)))
        elif self.bn == 'after':
            temp1 = self.bn1(F.relu(temp1.mm(self.pre_mats)))
            temp2 = self.bn2(F.relu(temp2.mm(self.pre_mats)))
        if len(embeds3) > 0:
            if self.bn == 'no':
                temp3 = F.relu(temp3.mm(self.pre_mats))
            elif self.bn == 'before':
                temp3 = F.relu(self.bn3(temp3.mm(self.pre_mats)))
            elif self.bn == 'after':
                temp3 = self.bn3(F.relu(temp3.mm(self.pre_mats)))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        # dim=0 means
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined


class MeanSet(nn.Module):
    def __init__(self):
        super(MeanSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.mean(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)
        else:
            return torch.mean(torch.stack([embeds1, embeds2], dim=0), dim=0)


class MinSet(nn.Module):
    def __init__(self):
        super(MinSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.min(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)[0]
        else:
            return torch.min(torch.stack([embeds1, embeds2], dim=0), dim=0)[0]


# this one is deepSet
class OffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, agg_func=torch.min, name='Real_offset'):
        super(OffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        if offset_use_center:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
            nn.init.xavier_uniform(self.pre_mats)
            self.register_parameter("premat_%s" % name, self.pre_mats)
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
            nn.init.xavier_uniform(self.pre_mats)
            self.register_parameter("premat_%s" % name, self.pre_mats)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s" % name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if self.offset_use_center:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3_o) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1_o
            temp2 = embeds2_o
            if len(embeds3_o) > 0:
                temp3 = embeds3_o
        temp1 = F.relu(temp1.mm(self.pre_mats))
        temp2 = F.relu(temp2.mm(self.pre_mats))
        if len(embeds3_o) > 0:
            temp3 = F.relu(temp3.mm(self.pre_mats))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        # agg_func is mean
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined


# this one is the offset in this paper
class InductiveOffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, off_reg, agg_func=torch.min, name='Real_offset'):
        super(InductiveOffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        self.off_reg = off_reg
        self.OffsetSet_Module = OffsetSet(mode_dims, expand_dims, offset_use_center, self.agg_func)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if len(embeds3_o) > 0:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o, embeds3_o]), dim=0)[0]
        else:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o]), dim=0)[0]
        offset = offset_min * F.sigmoid(
            self.OffsetSet_Module(embeds1, embeds1_o, embeds2, embeds2_o, embeds3, embeds3_o))
        return offset


# att_type = ele
# this is the center of this paper
class AttentionSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_reg=0., att_tem=1., att_type="whole", bn='no',
                 nat=1, name="Real"):
        super(AttentionSet, self).__init__()
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.att_type = att_type
        self.att_tem = att_tem
        self.Attention_module = Attention(mode_dims, expand_dims, center_use_offset, att_type=att_type, bn=bn, nat=nat)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        temp1 = (self.Attention_module(embeds1, embeds1_o) + self.att_reg) / (self.att_tem + 1e-4)
        temp2 = (self.Attention_module(embeds2, embeds2_o) + self.att_reg) / (self.att_tem + 1e-4)
        if len(embeds3) > 0:
            temp3 = (self.Attention_module(embeds3, embeds3_o) + self.att_reg) / (self.att_tem + 1e-4)
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat([temp1, temp2, temp3], dim=1), dim=1)
                center = embeds1 * (combined[:, 0].view(embeds1.size(0), 1)) + \
                         embeds2 * (combined[:, 1].view(embeds2.size(0), 1)) + \
                         embeds3 * (combined[:, 2].view(embeds3.size(0), 1))
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack([temp1, temp2, temp3]), dim=0)
                center = embeds1 * combined[0] + embeds2 * combined[1] + embeds3 * combined[2]
        else:
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat([temp1, temp2], dim=1), dim=1)
                center = embeds1 * (combined[:, 0].view(embeds1.size(0), 1)) + \
                         embeds2 * (combined[:, 1].view(embeds2.size(0), 1))
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack([temp1, temp2]), dim=0)
                center = embeds1 * combined[0] + embeds2 * combined[1]

        return center


# the MLP used in calculate center. More specifically, the MLP in a_i
class Attention(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_type, bn, nat, name="Real"):
        super(Attention, self).__init__()
        self.center_use_offset = center_use_offset
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
        else:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.atten_mats1)
        self.register_parameter("atten_mats1_%s" % name, self.atten_mats1)
        if self.nat >= 2:
            self.atten_mats1_1 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_1)
            self.register_parameter("atten_mats1_1_%s" % name, self.atten_mats1_1)
        if self.nat >= 3:
            self.atten_mats1_2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_2)
            self.register_parameter("atten_mats1_2_%s" % name, self.atten_mats1_2)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn1_1 = nn.BatchNorm1d(mode_dims)
            self.bn1_2 = nn.BatchNorm1d(mode_dims)
        if att_type == 'whole':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, 1))
        elif att_type == 'ele':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
        nn.init.xavier_uniform(self.atten_mats2)
        self.register_parameter("atten_mats2_%s" % name, self.atten_mats2)

    def forward(self, center_embed, offset_embed=None):
        if self.center_use_offset:
            temp1 = torch.cat([center_embed, offset_embed], dim=1)
        else:
            temp1 = center_embed
        if self.nat >= 1:
            if self.bn == 'no':
                temp2 = F.relu(temp1.mm(self.atten_mats1))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1(temp1.mm(self.atten_mats1)))
            elif self.bn == 'after':
                temp2 = self.bn1(F.relu(temp1.mm(self.atten_mats1)))
        if self.nat >= 2:
            if self.bn == 'no':
                temp2 = F.relu(temp2.mm(self.atten_mats1_1))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_1(temp2.mm(self.atten_mats1_1)))
            elif self.bn == 'after':
                temp2 = self.bn1_1(F.relu(temp2.mm(self.atten_mats1_1)))
        if self.nat >= 3:
            if self.bn == 'no':
                temp2 = F.relu(temp2.mm(self.atten_mats1_2))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_2(temp2.mm(self.atten_mats1_2)))
            elif self.bn == 'after':
                temp2 = self.bn1_2(F.relu(temp2.mm(self.atten_mats1_2)))
        temp3 = temp2.mm(self.atten_mats2)
        return temp3


class Query2box(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 writer=None, geo=None,
                 cen=None, offset_deepsets=None,
                 center_deepsets=None, offset_use_center=None, center_use_offset=None,
                 att_reg=0., off_reg=0., att_tem=1., euo=False,
                 gamma2=0, bn='no', nat=1, activation='relu'):
        super(Query2box, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.writer = writer
        self.geo = geo
        self.cen = cen
        self.offset_deepsets = offset_deepsets
        self.center_deepsets = center_deepsets
        self.offset_use_center = offset_use_center
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.off_reg = off_reg
        self.att_tem = att_tem
        self.euo = euo
        self.his_step = 0
        self.bn = bn
        self.nat = nat
        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        if gamma2 == 0:
            gamma2 = gamma

        self.gamma2 = nn.Parameter(
            torch.Tensor([gamma2]),
            requires_grad=False
        )
        # what's this?
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.entity_covariance = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_covariance,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_covariance = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_covariance,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )


        self.my_weight = nn.Parameter(torch.tensor([1]).float())

        if self.center_deepsets == 'vanilla':
            self.center_sets = CenterSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                         agg_func=torch.mean, bn=bn, nat=nat)
        elif self.center_deepsets == 'attention':
            self.center_sets = AttentionSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                            att_reg=self.att_reg, att_tem=self.att_tem, bn=bn, nat=nat)
        elif self.center_deepsets == 'eleattention':
            self.center_sets = AttentionSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                            att_reg=self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
        elif self.center_deepsets == 'mean':
            self.center_sets = MeanSet()
        else:
            assert False

        if self.offset_deepsets == 'vanilla':
            self.offset_sets = OffsetSet(self.relation_dim, self.relation_dim, self.offset_use_center,
                                         agg_func=torch.mean)
        elif self.offset_deepsets == 'inductive':
            self.offset_sets = InductiveOffsetSet(self.relation_dim, self.relation_dim, self.offset_use_center,
                                                  self.off_reg, agg_func=torch.mean)
        elif self.offset_deepsets == 'min':
            self.offset_sets = MinSet()
        else:
            assert False

        if model_name not in ['TransE', 'BoxTransE']:
            raise ValueError('model %s not supported' % model_name)

    # model((positive_sample, negative_sample), rel_len, qtype, mode=mode)
    # new_query_center = self.center_sets(query_center_1, offset_1, query_center_2, offset_2)
    # new_offset = self.offset_sets(query_center_1, offset_1, query_center_2, offset_2)
    def forward(self, sample, rel_len, qtype, mode='single'):
        tail_offset = None
        if qtype == 'chain-inter':
            assert mode == 'tail-batch'
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            # head_1 is positive, head_2 is negative
            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)
            if self.euo and self.geo == 'box':
                head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                   index=head_part[:, 0]).unsqueeze(1)
                head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                   index=head_part[:, 3]).unsqueeze(1)
                head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)

            relation_11 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            if self.geo == 'box':
                offset_11 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                relation_offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

        elif qtype == 'inter-chain' or qtype == 'union-chain':
            assert mode == 'tail-batch'
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)
            if self.euo and self.geo == 'box':
                head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                   index=head_part[:, 0]).unsqueeze(1)
                head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                   index=head_part[:, 2]).unsqueeze(1)
                head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                   negative_sample_size,
                                                                                                   -1)

            relation_11 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            if self.geo == 'box':
                offset_11 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                relation_offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

        elif qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                head_offset_1 = torch.index_select(self.entity_covariance, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_offset_2 = torch.index_select(self.entity_covariance, dim=0, index=sample[:, 2]).unsqueeze(1)
                head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)

                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                    head_offset_3 = torch.index_select(self.entity_covariance, dim=0, index=sample[:, 4]).unsqueeze(1)
                    head_offset = torch.cat([head_offset, head_offset_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                tail_offset = torch.index_select(self.entity_covariance, dim=0, index=sample[:, -1]).unsqueeze(1)

                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                    tail_offset = torch.cat([tail_offset, tail_offset], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)
                    tail_offset = torch.cat([tail_offset, tail_offset, tail_offset], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.relation_covariance, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_covariance, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation_offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.relation_covariance, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation_offset = torch.cat([relation_offset, offset_3], dim=0)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                head_offset_1 = torch.index_select(self.entity_covariance, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_offset_2 = torch.index_select(self.entity_covariance, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)

                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)

                    head_offset_3 = torch.index_select(self.entity_covariance, dim=0, index=head_part[:, 4]).unsqueeze(
                        1)
                    head_offset = torch.cat([head_offset, head_offset_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                tail_offset = torch.index_select(self.entity_covariance, dim=0, index=tail_part.view(-1)).view(
                    batch_size, negative_sample_size, -1)

                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                    tail_offset = torch.cat([tail_offset, tail_offset], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)
                    tail_offset = torch.cat([tail_offset, tail_offset, tail_offset], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                offset_1 = torch.index_select(self.relation_covariance, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.relation_covariance, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation_offset = torch.cat([offset_1, offset_2], dim=0)
                if rel_len == 3:
                    offset_3 = torch.index_select(self.relation_covariance, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation_offset = torch.cat([relation_offset, offset_3], dim=0)

        elif qtype == '1-chain' or qtype == '2-chain' or qtype == '3-chain':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1

                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_offset = torch.index_select(self.entity_covariance, dim=0, index=sample[:, 0]).unsqueeze(1)


                relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_offset = torch.index_select(self.relation_covariance, dim=0, index=sample[:, 1]).unsqueeze(
                        1).unsqueeze(1)

                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                    offset2 = torch.index_select(self.relation_covariance, dim=0, index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation_offset = torch.cat([relation_offset, offset2], 1)
                if rel_len == 3:
                    relation3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)
                    offset3 = torch.index_select(self.relation_covariance, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_offset = torch.cat([relation_offset, offset3], 1)

                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert relation_offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                tail_offset = torch.index_select(self.entity_covariance, dim=0, index=sample[:, -1]).unsqueeze(1)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_offset = torch.index_select(self.entity_covariance, dim=0, index=head_part[:, 0]).unsqueeze(1)


                relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_offset = torch.index_select(self.relation_covariance, dim=0, index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                    offset2 = torch.index_select(self.relation_covariance, dim=0, index=head_part[:, 2]).unsqueeze(
                            1).unsqueeze(1)
                    relation_offset = torch.cat([relation_offset, offset2], 1)
                if rel_len == 3:
                    relation3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)
                    offset3 = torch.index_select(self.relation_covariance, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_offset = torch.cat([relation_offset, offset3], 1)

                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert relation_offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                tail_offset = torch.index_select(self.entity_covariance, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'BoxTransE': self.BoxTransE
        }
        if self.geo == 'vec':
            relation_offset = None
            head_offset = None
        # if self.geo == 'box':
        #    if not self.euo:
        #        head_offset = None

        if self.model_name in model_func:
            if qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union':
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail,
                                                                                               mode, relation_offset,
                                                                                               head_offset, tail_offset,
                                                                                               1, qtype)
            else:
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail,
                                                                                               mode, relation_offset,
                                                                                               head_offset, tail_offset,
                                                                                               rel_len, qtype)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score, score_cen, offset_norm, score_cen_plus, None, None

    def gaussian_intersection(self, u1, u2, sigma1, sigma2):
        sigma1 = torch.exp(sigma1)
        sigma2 = torch.exp(sigma2)
        new_u = sigma2 * u1 / (sigma1 + sigma2) + sigma1 * u2 / (sigma1 + sigma2)
        new_v = 1 / ((1 / sigma1) + (1 / sigma2))
        return new_u, torch.log(new_v)

    def KL_diversion(self, mu_i, mu_j, sigma_i, sigma_j):
        """
        :param mu_i: mu of word i: [batch, embed]
        :param mu_j: mu of word j: [batch, embed]
        :param sigma_i: sigma of word i: [batch, embed]
        :param sigma_j: sigma of word j: [batch, embed]
        :return: the energy function between the two batchs of  data: [batch]
        """
        #sigma_i = torch.exp(sigma_i)
        #sigma_j = torch.exp(sigma_j)
        #assert mu_i.size()[0] == mu_j.size()[0]
        #a_ = torch.log(sigma_i)
        #a = torch.sum(a_, 2)
        #b = torch.sum(torch.log(sigma_j), 2)

        #det_fac = a - b
        #trace_fac = torch.sum(sigma_j / sigma_i, 2)
        #diff_mu = torch.sum((mu_i - mu_j) ** 2 / sigma_i, 2)
        #res =  0.5 * (trace_fac - det_fac + diff_mu - self.entity_dim)
        #return torch.log(res)
        distance = torch.norm(mu_i - mu_j, dim=2)
        return distance



    def BoxTransE(self, head, relation, tail, mode, relation_offset, head_offset, tail_offset, rel_len, qtype):

        query_mean = head
        for rel in range(rel_len):
            # the final mean
            query_mean = query_mean + relation[:, rel, :, :]

        # calculate covariance
        # if h, r, t, r_covariance = h_covariance + t_covariance
        t_covariance = head_offset
        for rel in range(0, rel_len):
            t_covariance = self.func(relation_offset[:, rel, :, :] - t_covariance)

        if 'inter' not in qtype and 'union' not in qtype:

            score_ = query_mean - tail
            offset_score_ = t_covariance - tail_offset
        else:

            rel_len = int(qtype.split('-')[0])
            assert rel_len > 1

            t_covariance = self.func(t_covariance)
            ts_covariance = torch.chunk(t_covariance, rel_len, dim=0)
            queries_center = torch.chunk(query_mean, rel_len, dim=0)

            tails = torch.chunk(tail, rel_len, dim=0)
            tails_offset = torch.chunk(tail_offset, rel_len, dim=0)

            if rel_len == 2:
                new_query_center = self.center_sets(queries_center[0].squeeze(1), ts_covariance[0].squeeze(1),
                                                    queries_center[1].squeeze(1), ts_covariance[1].squeeze(1))
                new_offset = self.offset_sets(queries_center[0].squeeze(1), ts_covariance[0].squeeze(1),
                                              queries_center[1].squeeze(1), ts_covariance[1].squeeze(1))

            elif rel_len == 3:
                new_query_center = self.center_sets(queries_center[0].squeeze(1), ts_covariance[0].squeeze(1),
                                                    queries_center[1].squeeze(1), ts_covariance[1].squeeze(1),
                                                    queries_center[2].squeeze(1), ts_covariance[2].squeeze(1))
                new_offset = self.offset_sets(queries_center[0].squeeze(1), ts_covariance[0].squeeze(1),
                                              queries_center[1].squeeze(1), ts_covariance[1].squeeze(1),
                                              queries_center[2].squeeze(1), ts_covariance[2].squeeze(1))
            #new_query_min = (new_query_center - 0.5 * self.func(new_offset)).unsqueeze(1)
            #new_query_max = (new_query_center + 0.5 * self.func(new_offset)).unsqueeze(1)
            #score_offset = F.relu(new_query_min - tails[0]) + F.relu(tails[0] - new_query_max)
            #score_center = new_query_center.unsqueeze(1) - tails[0]
            #score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tails[0])) - new_query_center.unsqueeze(1)

            #if rel_len == 2:
            #    new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
            #                                   queries_center[1].squeeze(1), offsets[1].squeeze(1))
            #    new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
            #                                   queries_center[1].squeeze(1), offsets[1].squeeze(1))

            #    new_query_center, new_offset = self.gaussian_intersection(queries_center[0], queries_center[1],
            #                                                              ts_covariance[0], ts_covariance[1])

            #elif rel_len == 3:
            #    new_query_center, new_offset = self.gaussian_intersection(queries_center[0], queries_center[1],
            #                                                              ts_covariance[0], ts_covariance[1])
            #    new_query_center, new_offset = self.gaussian_intersection(new_query_center, queries_center[2],
            #                                                              new_offset, ts_covariance[2])
            #aaaaa = new_query_center.unsqueeze(1)
            # for deepset
            # distance of the mean
            new_offset = self.func(new_offset)
            score_ = new_query_center.unsqueeze(1) - tails[0]
            offset_score_ =  tails_offset[0] - new_offset.unsqueeze(1)



        #score_ = self.KL_diversion(new_query_center.unsqueeze(1), tails[0], new_offset.unsqueeze(1), tails_offset[0])
        #score_ = self.KL_diversion(new_query_center, tails[0], new_offset, tails_offset[0])


        score = self.gamma.item() - torch.norm(score_, p=1, dim=-1) - torch.norm(offset_score_, p=1, dim=-1) * self.my_weight
        score_center = self.gamma2.item() - torch.norm(score_, p=1, dim=-1) - torch.norm(offset_score_, p=1, dim=-1) * self.my_weight
        score_center_plus = self.gamma.item() - torch.norm(score_, p=1, dim=-1) - torch.norm(offset_score_, p=1, dim=-1) * self.my_weight

        return score, score_center, torch.mean(
            torch.norm(relation_offset, p=2, dim=2).squeeze(1)), score_center_plus, None


    @staticmethod
    # log = query2box.train_step(query2box, optimizer, train_iterator_2i, args, step)
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        rel_len = int(train_iterator.qtype.split('-')[0])
        qtype = train_iterator.qtype
        _x, _x, _x, negative_score_cen_plus, _, _ = model(
            (positive_sample, negative_sample), rel_len, qtype, mode=mode)

        negative_score = F.logsigmoid(-negative_score_cen_plus).mean(dim=1)

        _x, _x, _x, positive_score_cen_plus, _, _ = model(positive_sample,
                                                                                                   rel_len, qtype)
        positive_score = F.logsigmoid(positive_score_cen_plus).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()
        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_triples, test_ans, test_ans_hard, args):
        qtype = test_triples[0][-1]
        rel_len = int(test_triples[0][-1].split('-')[0])

        model.eval()

        if qtype == 'inter-chain' or qtype == 'union-chain':
            test_dataloader_tail = DataLoader(
                TestInterChainDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        elif qtype == 'chain-inter':
            test_dataloader_tail = DataLoader(
                TestChainInterDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        elif 'inter' in qtype or 'union' in qtype:
            test_dataloader_tail = DataLoader(
                TestInterDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )
        else:
            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    test_ans,
                    test_ans_hard,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )

        test_dataset_list = [test_dataloader_tail]
        # test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        logs = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode, query in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    assert batch_size == 1, batch_size

                    if 'inter' in qtype:
                        if model.geo == 'box':
                            _, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len,
                                                                          qtype, mode=mode)
                        else:
                            score, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample),
                                                                              rel_len, qtype, mode=mode)
                    else:
                        score, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len,
                                                                          qtype, mode=mode)

                    if model.geo == 'box':
                        score = score_cen
                        score2 = score_cen_plus
                    # score = gamma - KL
                    # score is less (-100 < -10), KL is bigger, performance is worse
                    score -= (torch.min(score) - 1)
                    ans = test_ans[query]
                    hard_ans = test_ans_hard[query]
                    all_idx = set(range(args.nentity))
                    false_ans = all_idx - ans
                    ans_list = list(ans)
                    hard_ans_list = list(hard_ans)
                    false_ans_list = list(false_ans)
                    ans_idxs = np.array(hard_ans_list)
                    vals = np.zeros((len(ans_idxs), args.nentity))
                    vals[np.arange(len(ans_idxs)), ans_idxs] = 1
                    axis2 = np.tile(false_ans_list, len(ans_idxs))
                    axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
                    vals[axis1, axis2] = 1
                    b = torch.Tensor(vals) if not args.cuda else torch.Tensor(vals).cuda()
                    filter_score = b * score
                    # Returns the indices that sort a tensor along a given dimension in ascending
                    # order by value.
                    argsort = torch.argsort(filter_score, dim=1, descending=True)
                    ans_tensor = torch.LongTensor(hard_ans_list) if not args.cuda else torch.LongTensor(
                        hard_ans_list).cuda()
                    argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
                    ranking = (argsort == 0).nonzero()
                    ranking = ranking[:, 1]
                    ranking = ranking + 1
                    if model.geo == 'box':
                        score2 -= (torch.min(score2) - 1)
                        filter_score2 = b * score2
                        argsort2 = torch.argsort(filter_score2, dim=1, descending=True)
                        argsort2 = torch.transpose(torch.transpose(argsort2, 0, 1) - ans_tensor, 0, 1)
                        ranking2 = (argsort2 == 0).nonzero()
                        ranking2 = ranking2[:, 1]
                        ranking2 = ranking2 + 1

                    ans_vec = np.zeros(args.nentity)
                    ans_vec[ans_list] = 1
                    hits1 = torch.sum((ranking <= 1).to(torch.float)).item()
                    hits3 = torch.sum((ranking <= 3).to(torch.float)).item()
                    hits10 = torch.sum((ranking <= 10).to(torch.float)).item()
                    mr = float(torch.sum(ranking).item())
                    mrr = torch.sum(1. / ranking.to(torch.float)).item()
                    hits1m = torch.mean((ranking <= 1).to(torch.float)).item()
                    hits3m = torch.mean((ranking <= 3).to(torch.float)).item()
                    hits10m = torch.mean((ranking <= 10).to(torch.float)).item()
                    mrm = torch.mean(ranking.to(torch.float)).item()
                    mrrm = torch.mean(1. / ranking.to(torch.float)).item()
                    num_ans = len(hard_ans_list)
                    if model.geo == 'box':
                        hits1m_newd = torch.mean((ranking2 <= 1).to(torch.float)).item()
                        hits3m_newd = torch.mean((ranking2 <= 3).to(torch.float)).item()
                        hits10m_newd = torch.mean((ranking2 <= 10).to(torch.float)).item()
                        mrm_newd = torch.mean(ranking2.to(torch.float)).item()
                        mrrm_newd = torch.mean(1. / ranking2.to(torch.float)).item()
                    else:
                        hits1m_newd = hits1m
                        hits3m_newd = hits3m
                        hits10m_newd = hits10m
                        mrm_newd = mrm
                        mrrm_newd = mrrm

                    logs.append({
                        'MRRm_new': mrrm_newd,
                        'MRm_new': mrm_newd,
                        'HITS@1m_new': hits1m_newd,
                        'HITS@3m_new': hits3m_newd,
                        'HITS@10m_new': hits10m_newd,
                        'num_answer': num_ans
                    })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        num_answer = sum([log['num_answer'] for log in logs])
        for metric in logs[0].keys():
            if metric == 'num_answer':
                continue
            if 'm' in metric:
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            else:
                metrics[metric] = sum([log[metric] for log in logs]) / num_answer
        return metrics