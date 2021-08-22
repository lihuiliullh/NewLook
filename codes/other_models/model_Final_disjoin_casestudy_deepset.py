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
import time


def Identity(x):
    return x


class MultilayerNN(nn.Module):
    # torch min will just return a single element in the tensor
    def __init__(self, center_dim, offset_dim):
        super(MultilayerNN, self).__init__()
        self.center_dim = center_dim
        self.offset_dim = offset_dim

        assert center_dim == offset_dim

        #
        expand_dim = center_dim * 2

        self.mats1_center = nn.Parameter(torch.FloatTensor(center_dim,  expand_dim))
        nn.init.xavier_uniform(self.mats1_center)
        self.register_parameter("mats1_center", self.mats1_center)

        self.mats1_offset = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform(self.mats1_offset)
        self.register_parameter("mats1_offset", self.mats1_offset)

        self.mats1_offset2 = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform(self.mats1_offset2)
        self.register_parameter("mats1_offset2", self.mats1_offset2)

        self.post_mats_center = nn.Parameter(torch.FloatTensor(expand_dim * 3, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform(self.post_mats_center)
        self.register_parameter("post_mats_center", self.post_mats_center)

        self.post_mats_offset = nn.Parameter(torch.FloatTensor(expand_dim * 3, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform(self.post_mats_offset)
        self.register_parameter("post_mats_offset", self.post_mats_offset)

        self.post_mats_offset2 = nn.Parameter(torch.FloatTensor(expand_dim * 3, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform(self.post_mats_offset2)
        self.register_parameter("post_mats_offset2", self.post_mats_offset2)

    def forward(self, center_emb, query_min_emb, query_max_emb):
        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        temp1 = F.relu(torch.matmul(center_emb, self.mats1_center))
        temp2 = F.relu(torch.matmul(query_min_emb, self.mats1_offset))
        temp3 = F.relu(torch.matmul(query_max_emb, self.mats1_offset2))

        temp4 = torch.cat([temp1, temp2, temp3], dim=2)

        out_center = torch.matmul(temp4, self.post_mats_center)
        out_min = torch.matmul(temp4, self.post_mats_offset)
        out_max = torch.matmul(temp4, self.post_mats_offset2)
        return (out_center, out_min, out_max)






class DisjoinNN(nn.Module):
    # torch min will just return a single element in the tensor
    def __init__(self, center_dim, offset_dim):
        super(DisjoinNN, self).__init__()
        self.center_dim = center_dim
        self.offset_dim = offset_dim

        assert center_dim == offset_dim

        expand_dim = center_dim

        self.mats1_center = nn.Parameter(torch.FloatTensor(center_dim * 2,  expand_dim))
        nn.init.xavier_uniform(self.mats1_center)
        self.register_parameter("mats1_center", self.mats1_center)

        #self.mats2_center = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        #nn.init.xavier_uniform(self.mats2_center)
        #self.register_parameter("mats2_center", self.mats2_center)

        self.mats1_offset = nn.Parameter(torch.FloatTensor(center_dim * 2, expand_dim))
        nn.init.xavier_uniform(self.mats1_offset)
        self.register_parameter("mats1_offset", self.mats1_offset)

        #self.mats1_offset2 = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        #nn.init.xavier_uniform(self.mats1_offset2)
        #self.register_parameter("mats1_offset2", self.mats1_offset2)

        self.post_mats_center = nn.Parameter(torch.FloatTensor(expand_dim, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform(self.post_mats_center)
        self.register_parameter("post_mats_center", self.post_mats_center)

        self.post_mats_offset = nn.Parameter(torch.FloatTensor(expand_dim, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform(self.post_mats_offset)
        self.register_parameter("post_mats_offset", self.post_mats_offset)

        self.post_mats_offset2 = nn.Parameter(torch.FloatTensor(expand_dim, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform(self.post_mats_offset2)
        self.register_parameter("post_mats_offset2", self.post_mats_offset2)


    def forward(self, center_emb1, min1, max1, center_emb2, min2, max2):
        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        # center, min, max should all use attention.
        offset1 = max1 - min1
        offset2 = max2 - min2
        a1 = torch.cat([center_emb1, offset1], dim=2)
        temp1 = F.relu(torch.matmul(a1, self.mats1_center))
        a2 = torch.cat([center_emb2, offset2], dim=2)
        temp2 = F.relu(torch.matmul(a2, self.mats1_offset))

        temp4 = temp1 + temp2

        out_center = torch.matmul(temp4, self.post_mats_center)
        out_min = torch.matmul(temp4, self.post_mats_offset)
        out_max = torch.matmul(temp4, self.post_mats_offset2)
        return (out_center, out_min, out_max)



####################################################################################################

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

####################################################################################################
# this one is deepSet
class DeepSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, agg_func=torch.min, name='Real_offset'):
        super(DeepSet, self).__init__()
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
class BoxOffsetNetwork(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, off_reg, agg_func=torch.min, name='Real_offset'):
        super(BoxOffsetNetwork, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        self.off_reg = off_reg
        self.OffsetSet_Module = DeepSet(mode_dims, expand_dims, offset_use_center, self.agg_func)

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
class CenterNetwork_AttentionSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_reg=0., att_tem=1., att_type="whole", bn='no',
                 nat=1, name="Real"):
        super(CenterNetwork_AttentionSet, self).__init__()
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.att_type = att_type
        self.att_tem = att_tem
        self.MLP_module = MLP(mode_dims, expand_dims, center_use_offset, att_type=att_type, bn=bn, nat=nat)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        temp1 = (self.MLP_module(embeds1, embeds1_o) + self.att_reg) / (self.att_tem + 1e-4)
        temp2 = (self.MLP_module(embeds2, embeds2_o) + self.att_reg) / (self.att_tem + 1e-4)
        if len(embeds3) > 0:
            temp3 = (self.MLP_module(embeds3, embeds3_o) + self.att_reg) / (self.att_tem + 1e-4)
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
class MLP(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_type, bn, nat, name="Real"):
        super(MLP, self).__init__()
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

        print("Model_box_neural")

        # -------
        self.center_trans = MultilayerNN(hidden_dim, hidden_dim)
        self.disjoin_nn = DisjoinNN(hidden_dim, hidden_dim)
        # -------

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

        # -------- shift begin
        # this dim should have the same size as embedding
        self.shift_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
        nn.init.uniform_(
            tensor=self.shift_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # -------- shift end

        if self.geo == 'vec':
            if self.center_deepsets == 'vanilla':
                self.transE_deepsets = CenterSet(self.relation_dim, self.relation_dim, False, agg_func=torch.mean, bn=bn,
                                                 nat=nat)
            elif self.center_deepsets == 'attention':
                self.transE_deepsets = CenterNetwork_AttentionSet(self.relation_dim, self.relation_dim, False,
                                                                  att_reg=self.att_reg, att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'eleattention':
                self.transE_deepsets = CenterNetwork_AttentionSet(self.relation_dim, self.relation_dim, False,
                                                                  att_reg=self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'mean':
                self.transE_deepsets = MeanSet()
            else:
                assert False

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding,
                a=0.,
                b=self.embedding_range.item()
            )
            if self.euo:
                self.entity_offset_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
                nn.init.uniform_(
                    tensor=self.entity_offset_embedding,
                    a=0.,
                    b=self.embedding_range.item()
                )

            # center_deepsets = eleattention
            if self.center_deepsets == 'vanilla':
                self.center_sets = CenterSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                             agg_func=torch.mean, bn=bn, nat=nat)
            elif self.center_deepsets == 'attention':
                self.center_sets = CenterNetwork_AttentionSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                                              att_reg=self.att_reg, att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'eleattention':
                self.center_sets = CenterNetwork_AttentionSet(self.relation_dim, self.relation_dim, self.center_use_offset,
                                                              att_reg=self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn,
                                                              nat=nat)
            elif self.center_deepsets == 'mean':
                self.center_sets = MeanSet()
            else:
                assert False
            # offset_deepsets = inductive
            if self.offset_deepsets == 'vanilla':
                self.offset_sets = DeepSet(self.relation_dim, self.relation_dim, self.offset_use_center,
                                           agg_func=torch.mean)
            elif self.offset_deepsets == 'inductive':
                self.offset_sets = BoxOffsetNetwork(self.relation_dim, self.relation_dim, self.offset_use_center,
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
    def forward(self, sample=None, rel_len=None, qtype=None, mode='single', find_subgraph=False, logical_triples=None, subgraph_info_map=None):
        if find_subgraph:
            # iterate each triple in the logical_triples
            result_map = {}
            for element in logical_triples:
                operation = element[0]
                if operation == 'anchor':
                    all_anchors = element[1]
                    for n in all_anchors:
                        idx = subgraph_info_map[n]
                        embed = torch.index_select(self.entity_embedding, dim=0,
                                                   index=torch.LongTensor([int(idx)]).cuda()).unsqueeze(1)
                        result_map[n] = {}
                        result_map[n]['center'] = embed
                elif operation == 'projection':
                    from_id = element[1]
                    predicate_id = element[2]
                    predicate_id = subgraph_info_map[predicate_id]
                    result_id = element[3]
                    if from_id not in result_map:
                        raise ValueError('key %s not in result_map' % str(from_id))
                    query_center = result_map[from_id]['center']
                    rel_embed = torch.index_select(self.relation_embedding, dim=0,
                                                   index=torch.LongTensor([int(predicate_id)]).cuda()).unsqueeze(1)
                    query_center = query_center + rel_embed
                    offset_embed = torch.index_select(self.offset_embedding, dim=0,
                                                      index=torch.LongTensor([int(predicate_id)]).cuda()).unsqueeze(1)
                    query_min = query_center - 0.5 * self.func(offset_embed)
                    query_max = query_center + 0.5 * self.func(offset_embed)
                    query_center, query_min, query_max = self.center_trans(query_center, query_min, query_max)
                    result_map[result_id] = {}
                    result_map[result_id]['center'] = query_center
                    result_map[result_id]['query_min'] = query_min
                    result_map[result_id]['query_max'] = query_max
                    result_map[result_id]['in_predicates'] = [predicate_id]
                elif operation == 'intersection':
                    # only two intersection or three intersection
                    data = element[1]
                    result_id = element[2]
                    if len(data) == 2:
                        head_id1 = data[0][0]
                        rel_id1 = data[0][1]
                        rel_id1 = subgraph_info_map[rel_id1]
                        head_id2 = data[1][0]
                        rel_id2 = data[1][1]
                        rel_id2 = subgraph_info_map[rel_id2]
                        center_embed1 = result_map[head_id1]['center']
                        center_embed2 = result_map[head_id2]['center']
                        rel_embed1 = torch.index_select(self.relation_embedding, dim=0,
                                                        index=torch.LongTensor([int(rel_id1)]).cuda()).unsqueeze(1)
                        rel_embed2 = torch.index_select(self.relation_embedding, dim=0,
                                                        index=torch.LongTensor([int(rel_id2)]).cuda()).unsqueeze(1)
                        offset_embed1 = torch.index_select(self.offset_embedding, dim=0,
                                                           index=torch.LongTensor([int(rel_id1)]).cuda()).unsqueeze(1)
                        offset_embed2 = torch.index_select(self.offset_embedding, dim=0,
                                                           index=torch.LongTensor([int(rel_id2)]).cuda()).unsqueeze(1)
                        # ----------
                        center_embed1 = center_embed1 + rel_embed1
                        query_min1 = center_embed1 - 0.5 * self.func(offset_embed1)
                        query_max1 = center_embed1 + 0.5 * self.func(offset_embed1)
                        query_center1, query_min1, query_max1 = self.center_trans(center_embed1, query_min1, query_max1)
                        offset1 = query_max1 - query_min1

                        center_embed2 = center_embed2 + rel_embed2
                        query_min2 = center_embed2 - 0.5 * self.func(offset_embed2)
                        query_max2 = center_embed2 + 0.5 * self.func(offset_embed2)
                        query_center2, query_min2, query_max2 = self.center_trans(center_embed2, query_min2, query_max2)
                        offset2 = query_max2 - query_min2

                        new_query_center = self.center_sets(query_center1.squeeze(1), offset1.squeeze(1),
                                                            query_center2.squeeze(1), offset2.squeeze(1)).unsqueeze(1)
                        new_offset = self.offset_sets(query_center1.squeeze(1), offset1.squeeze(1),
                                                      query_center2.squeeze(1), offset2.squeeze(1)).unsqueeze(1)

                        new_query_min = (new_query_center - 0.5 * self.func(new_offset))
                        new_query_max = (new_query_center + 0.5 * self.func(new_offset))

                        result_map[result_id] = {}
                        result_map[result_id]['center'] = new_query_center
                        result_map[result_id]['query_min'] = new_query_min
                        result_map[result_id]['query_max'] = new_query_max
                        result_map[result_id]['in_predicates'] = [rel_id1, rel_id2]
                    elif len(data) == 3:
                        head_id1 = data[0][0]
                        rel_id1 = data[0][1]
                        rel_id1 = subgraph_info_map[rel_id1]
                        head_id2 = data[1][0]
                        rel_id2 = data[1][1]
                        rel_id2 = subgraph_info_map[rel_id2]
                        head_id3 = data[2][0]
                        rel_id3 = data[2][1]
                        rel_id3 = subgraph_info_map[rel_id3]
                        center_embed1 = result_map[head_id1]['center']
                        center_embed2 = result_map[head_id2]['center']
                        center_embed3 = result_map[head_id3]['center']
                        rel_embed1 = torch.index_select(self.relation_embedding, dim=0,
                                                        index=torch.LongTensor([int(rel_id1)]).cuda()).unsqueeze(1)
                        rel_embed2 = torch.index_select(self.relation_embedding, dim=0,
                                                        index=torch.LongTensor([int(rel_id2)]).cuda()).unsqueeze(1)
                        rel_embed3 = torch.index_select(self.relation_embedding, dim=0,
                                                        index=torch.LongTensor([int(rel_id3)]).cuda()).unsqueeze(1)
                        offset_embed1 = torch.index_select(self.offset_embedding, dim=0,
                                                           index=torch.LongTensor([int(rel_id1)]).cuda()).unsqueeze(1)
                        offset_embed2 = torch.index_select(self.offset_embedding, dim=0,
                                                           index=torch.LongTensor([int(rel_id2)]).cuda()).unsqueeze(1)
                        offset_embed3 = torch.index_select(self.offset_embedding, dim=0,
                                                           index=torch.LongTensor([int(rel_id3)]).cuda()).unsqueeze(1)
                        # ----------
                        center_embed1 = center_embed1 + rel_embed1
                        query_min1 = center_embed1 - 0.5 * self.func(offset_embed1)
                        query_max1 = center_embed1 + 0.5 * self.func(offset_embed1)
                        query_center1, query_min1, query_max1 = self.center_trans(center_embed1, query_min1, query_max1)
                        offset1 = query_max1 - query_min1

                        center_embed2 = center_embed2 + rel_embed2
                        query_min2 = center_embed2 - 0.5 * self.func(offset_embed2)
                        query_max2 = center_embed2 + 0.5 * self.func(offset_embed2)
                        query_center2, query_min2, query_max2 = self.center_trans(center_embed2, query_min2, query_max2)
                        offset2 = query_max2 - query_min2

                        center_embed3 = center_embed3 + rel_embed3
                        query_min3 = center_embed3 - 0.5 * self.func(offset_embed3)
                        query_max3 = center_embed3 + 0.5 * self.func(offset_embed3)
                        query_center3, query_min3, query_max3 = self.center_trans(center_embed3, query_min3, query_max3)
                        offset3 = query_max3 - query_min3

                        new_query_center = self.center_sets(query_center1.squeeze(1), offset1.squeeze(1),
                                                            query_center2.squeeze(1), offset2.squeeze(1),
                                                            query_center3.squeeze(1), offset3.squeeze(1)).unsqueeze(1)
                        new_offset = self.offset_sets(query_center1.squeeze(1), offset1.squeeze(1),
                                                      query_center2.squeeze(1), offset2.squeeze(1),
                                                      query_center3.squeeze(1), offset3.squeeze(1)).unsqueeze(1)

                        new_query_min = (new_query_center - 0.5 * self.func(new_offset))
                        new_query_max = (new_query_center + 0.5 * self.func(new_offset))

                        result_map[result_id] = {}
                        result_map[result_id]['center'] = new_query_center
                        result_map[result_id]['query_min'] = new_query_min
                        result_map[result_id]['query_max'] = new_query_max
                        result_map[result_id]['in_predicates'] = [rel_id1, rel_id2, rel_id3]

                elif operation == 'disjoin':
                    head_id1 = element[1]
                    rel_id1 = element[2]
                    rel_id1 = subgraph_info_map[rel_id1]
                    head_id2 = element[3]
                    rel_id2 = element[4]
                    rel_id2 = subgraph_info_map[rel_id2]
                    result_id = element[5]
                    center_embed1 = result_map[head_id1]['center']
                    center_embed2 = result_map[head_id2]['center']
                    rel_embed1 = torch.index_select(self.relation_embedding, dim=0,
                                                    index=torch.LongTensor([int(rel_id1)]).cuda()).unsqueeze(1)
                    rel_embed2 = torch.index_select(self.relation_embedding, dim=0,
                                                    index=torch.LongTensor([int(rel_id2)]).cuda()).unsqueeze(1)
                    offset_embed1 = torch.index_select(self.offset_embedding, dim=0,
                                                       index=torch.LongTensor([int(rel_id1)]).cuda()).unsqueeze(1)
                    offset_embed2 = torch.index_select(self.offset_embedding, dim=0,
                                                       index=torch.LongTensor([int(rel_id2)]).cuda()).unsqueeze(1)

                    center_embed1 = center_embed1 + rel_embed1
                    query_min1 = center_embed1 - 0.5 * self.func(offset_embed1)
                    query_max1 = center_embed1 + 0.5 * self.func(offset_embed1)
                    query_center1, query_min1, query_max1 = self.center_trans(center_embed1, query_min1, query_max1)
                    offset1 = query_max1 - query_min1

                    center_embed2 = center_embed2 + rel_embed2
                    query_min2 = center_embed2 - 0.5 * self.func(offset_embed2)
                    query_max2 = center_embed2 + 0.5 * self.func(offset_embed2)
                    query_center2, query_min2, query_max2 = self.center_trans(center_embed2, query_min2, query_max2)
                    offset2 = query_max2 - query_min2

                    new_query_center, new_query_min, new_query_max = self.disjoin_nn(center_embed1, query_min1, query_max1,
                                                                                     center_embed2, query_min2, query_max2)
                    result_map[result_id] = {}
                    result_map[result_id]['center'] = new_query_center
                    result_map[result_id]['query_min'] = new_query_min
                    result_map[result_id]['query_max'] = new_query_max
                    result_map[result_id]['in_predicates'] = [rel_id1, rel_id2]
                elif operation == 'union':
                    head_id1 = element[1]
                    rel_id1 = element[2]
                    rel_id1 = subgraph_info_map[rel_id1]
                    head_id2 = element[3]
                    rel_id2 = element[4]
                    rel_id2 = subgraph_info_map[rel_id2]
                    result_id = element[5]

                    center_embed1 = result_map[head_id1]['center']
                    center_embed2 = result_map[head_id2]['center']
                    rel_embed1 = torch.index_select(self.relation_embedding, dim=0,
                                                    index=torch.LongTensor([int(rel_id1)]).cuda()).unsqueeze(1)
                    rel_embed2 = torch.index_select(self.relation_embedding, dim=0,
                                                    index=torch.LongTensor([int(rel_id2)]).cuda()).unsqueeze(1)
                    offset_embed1 = torch.index_select(self.offset_embedding, dim=0,
                                                       index=torch.LongTensor([int(rel_id1)]).cuda()).unsqueeze(1)
                    offset_embed2 = torch.index_select(self.offset_embedding, dim=0,
                                                       index=torch.LongTensor([int(rel_id2)]).cuda()).unsqueeze(1)

                    center_embed1 = center_embed1 + rel_embed1
                    query_min1 = center_embed1 - 0.5 * self.func(offset_embed1)
                    query_max1 = center_embed1 + 0.5 * self.func(offset_embed1)
                    query_center1, query_min1, query_max1 = self.center_trans(center_embed1, query_min1, query_max1)

                    center_embed2 = center_embed2 + rel_embed2
                    query_min2 = center_embed2 - 0.5 * self.func(offset_embed2)
                    query_max2 = center_embed2 + 0.5 * self.func(offset_embed2)
                    query_center2, query_min2, query_max2 = self.center_trans(center_embed2, query_min2, query_max2)

                    result_map[result_id] = {}
                    result_map[result_id]['center'] = [query_center1, query_center2]
                    result_map[result_id]['query_min'] = [query_min1, query_min2]
                    result_map[result_id]['query_max'] = [query_max1, query_max2]
                    result_map[result_id]['in_predicates'] = [rel_id1, rel_id2]

            return result_map


        shift_of_node = None

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

            shift1 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                1)
            shift2 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                1)
            shift_of_node = torch.cat([shift1, shift2], dim=0)


            if self.geo == 'box':
                offset_11 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

        elif qtype == 'inter-chain' or qtype == 'union-chain' or qtype == 'disjoin-chain':
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

            # -------
            shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
            # shift_of_node = torch.cat([shift_1], dim=0)
            # -------

            if self.geo == 'box':
                offset_11 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                offset_12 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(
                    1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

        elif qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union' or qtype == '2-disjoin' or qtype == '3-disjoin':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)
                if self.euo and self.geo == 'box':
                    head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                       index=sample[:, 0]).unsqueeze(1)
                    head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                       index=sample[:, 2]).unsqueeze(1)
                    head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)
                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)
                    if self.euo and self.geo == 'box':
                        head_offset_3 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                           index=sample[:, 4]).unsqueeze(1)
                        head_offset = torch.cat([head_offset, head_offset_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                if self.geo == 'box':
                    offset_1 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    offset_2 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset_1, offset_2], dim=0)
                    if rel_len == 3:
                        offset_3 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 5]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset_3], dim=0)

                shift_of_node1 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
                shift_of_node2 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 3]).unsqueeze(1)
                shift_of_node = torch.cat([shift_of_node1, shift_of_node2], dim=0)
                if rel_len == 3:
                    shift_of_node3 = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 5]).unsqueeze(1)
                    shift_of_node = torch.cat([shift_of_node, shift_of_node3], dim=0)


            elif mode == 'tail-batch':
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
                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)
                    if self.euo and self.geo == 'box':
                        head_offset_3 = torch.index_select(self.entity_offset_embedding, dim=0,
                                                           index=head_part[:, 4]).unsqueeze(1)
                        head_offset = torch.cat([head_offset, head_offset_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                if self.geo == 'box':
                    offset_1 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    offset = torch.cat([offset_1, offset_2], dim=0)
                    if rel_len == 3:
                        offset_3 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 5]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset_3], dim=0)

                shift_of_node1 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
                shift_of_node2 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                shift_of_node = torch.cat([shift_of_node1, shift_of_node2], dim=0)
                if rel_len == 3:
                    shift_of_node3 = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 5]).unsqueeze(1)
                    shift_of_node = torch.cat([shift_of_node, shift_of_node3], dim=0)

        elif qtype == '1-chain' or qtype == '2-chain' or qtype == '3-chain':
            if mode == 'single':
                # just one negative or positive sample
                batch_size, negative_sample_size = sample.size(0), 1

                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)

                relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)


                if self.geo == 'box':
                    offset = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    if self.euo:
                        head_offset = torch.index_select(self.entity_offset_embedding, dim=0,
                                                         index=sample[:, 0]).unsqueeze(1)
                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)

                    if self.geo == 'box':
                        offset2 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset2], 1)
                if rel_len == 3:
                    relation3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)

                    if self.geo == 'box':
                        offset3 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset3], 1)

                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)

                # ------ get shift, only one shift for chain in the end
                if rel_len == 3:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 3]).unsqueeze(
                        1)
                elif rel_len == 2:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 2]).unsqueeze(
                        1)
                elif rel_len == 1:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=sample[:, 1]).unsqueeze(
                        1)


            elif mode == 'tail-batch':
                # batch size
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)

                relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                if self.geo == 'box':
                    offset = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(
                        1).unsqueeze(1)
                    if self.euo:
                        head_offset = torch.index_select(self.entity_offset_embedding, dim=0,
                                                         index=head_part[:, 0]).unsqueeze(1)
                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)

                    if self.geo == 'box':
                        offset2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset2], 1)
                if rel_len == 3:
                    relation3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)

                    if self.geo == 'box':
                        offset3 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(
                            1).unsqueeze(1)
                        offset = torch.cat([offset, offset3], 1)

                # ------ get shift, only one shift for chain in the end
                if rel_len == 3:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                elif rel_len == 2:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                elif rel_len == 1:
                    shift_of_node = torch.index_select(self.shift_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)


                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size,
                                                                                                       -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'BoxTransE': self.BoxTransE,
            'TransE': self.TransE,
        }
        if self.geo == 'vec':
            offset = None
            head_offset = None
        if self.geo == 'box':
            if not self.euo:
                head_offset = None

        if shift_of_node is None:
            break_here = True

        if self.model_name in model_func:
            if qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union' or qtype == '2-disjoin' or qtype == '3-disjoin':
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail,
                                                                                               mode, offset,
                                                                                               head_offset, 1, qtype, shift_of_node)
            else:
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail,
                                                                                               mode, offset,
                                                                                               head_offset, rel_len,
                                                                                               qtype, shift_of_node)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score, score_cen, offset_norm, score_cen_plus, None, None

    def BoxTransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype, shift_of_node):

        if qtype == 'chain-inter':
            # chain after intersection
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)

            heads = torch.chunk(head, 2, dim=0)

            #query_center_1 = heads[0] + relations[0][:, 0, :, :] + relations[1][:, 0, :, :]
            #query_center_2 = heads[1] + relations[2][:, 0, :, :]

            if self.euo:
                query_center_1 = heads[0]
                query_min_1 = query_center_1 - 0.5 * self.func(head_offsets[0])
                query_max_1 = query_center_1 + 0.5 * self.func(head_offsets[0])
                for i in range(2):
                    query_center_1 = query_center_1 + relations[i][:, 0, :, :]
                    query_min_1 = query_center_1 - 0.5 * self.func(offsets[i][:, 0, :, :])
                    query_max_1 = query_center_1 + 0.5 * self.func(offsets[i][:, 0, :, :])
                    # put it through a neural network
                    query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1, query_max_1)

                query_center_2 = heads[1] + relations[2][:, 0, :, :]
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[2][:, 0, :, :]) # - 0.5 * self.func(head_offsets[1])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[2][:, 0, :, :]) # + 0.5 * self.func(head_offsets[1])
                query_center_2, query_min_2, query_max_2 = self.center_trans(query_center_2, query_min_2, query_max_2)
            else:
                query_center_1 = heads[0]
                query_min_1 = query_center_1
                query_max_1 = query_center_1
                for i in range(2):
                    query_center_1 = query_center_1 + relations[i][:, 0, :, :]
                    query_min_1 = query_center_1 - 0.5 * self.func(offsets[i][:, 0, :, :])
                    query_max_1 = query_center_1 + 0.5 * self.func(offsets[i][:, 0, :, :])
                    query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1,
                                                                                 query_max_1)

                query_center_2 = heads[1] + relations[2][:, 0, :, :]
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[2][:, 0, :, :])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[2][:, 0, :, :])
                query_center_2, query_min_2, query_max_2 = self.center_trans(query_center_2, query_min_2, query_max_2)

            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            offset_1 = (query_max_1 - query_min_1).squeeze(1)
            offset_2 = (query_max_2 - query_min_2).squeeze(1)
            new_query_center = self.center_sets(query_center_1, offset_1, query_center_2, offset_2)
            new_offset = self.offset_sets(query_center_1, offset_1, query_center_2, offset_2)
            new_query_min = (new_query_center - 0.5 * self.func(new_offset)).unsqueeze(1)
            new_query_max = (new_query_center + 0.5 * self.func(new_offset)).unsqueeze(1)

            shifts = torch.chunk(shift_of_node, 2, dim=0)
            shift = (shifts[0] + shifts[1]) / 2


            score_offset = F.relu(new_query_min - tail - shift) + F.relu(tail + shift - new_query_max)
            score_center = new_query_center.unsqueeze(1) - tail - shift
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail + shift)) - new_query_center.unsqueeze(1)


        elif qtype == 'inter-chain' or qtype == 'disjoin-chain':
            # intersection after chain
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)

            heads = torch.chunk(head, 2, dim=0)

            query_center_1 = heads[0] + relations[0][:, 0, :, :]
            query_center_2 = heads[1] + relations[1][:, 0, :, :]
            if self.euo:
                query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:, 0, :, :]) #- 0.5 * self.func(head_offsets[0])
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[1][:, 0, :, :]) #- 0.5 * self.func(head_offsets[1])
                query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:, 0, :, :]) #+ 0.5 * self.func(head_offsets[0])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[1][:, 0, :, :]) #+ 0.5 * self.func(head_offsets[1])
            else:
                query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:, 0, :, :])
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[1][:, 0, :, :])
                query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:, 0, :, :])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[1][:, 0, :, :])

            query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1, query_max_1)
            query_center_2, query_min_2, query_max_2 = self.center_trans(query_center_2, query_min_2, query_max_2)

            if qtype == 'inter-chain':
                query_center_1 = query_center_1.squeeze(1)
                query_center_2 = query_center_2.squeeze(1)
                offset_1 = (query_max_1 - query_min_1).squeeze(1)
                offset_2 = (query_max_2 - query_min_2).squeeze(1)
                conj_query_center = self.center_sets(query_center_1, offset_1, query_center_2, offset_2).unsqueeze(1)
                new_offset = self.offset_sets(query_center_1, offset_1, query_center_2, offset_2).unsqueeze(1)
                new_query_center = conj_query_center + relations[2][:, 0, :, :]

                new_query_min = new_query_center - 0.5 * self.func(offsets[2][:, 0, :, :]) #- 0.5 * self.func(new_offset)
                new_query_max = new_query_center + 0.5 * self.func(offsets[2][:, 0, :, :]) #+ 0.5 * self.func(new_offset)

                new_query_center, new_query_min, new_query_max = self.center_trans(new_query_center, new_query_min, new_query_max)
            elif qtype == 'disjoin-chain':
                query_center_1 = query_center_1
                query_center_2 = query_center_2
                offset_1 = (query_max_1 - query_min_1)
                offset_2 = (query_max_2 - query_min_2)
                disjoin_center, new_min, new_max = self.disjoin_nn(query_center_1, query_min_1, query_max_1, query_center_2, query_min_2, query_max_2)

                new_query_center = disjoin_center + relations[2][:, 0, :, :]
                new_query_min = new_query_center - 0.5 * self.func(offsets[2][:, 0, :, :])  # - 0.5 * self.func(new_offset)
                new_query_max = new_query_center + 0.5 * self.func(offsets[2][:, 0, :, :])  # + 0.5 * self.func(new_offset)

                new_query_center, new_query_min, new_query_max = self.center_trans(new_query_center, new_query_min, new_query_max)

            score_offset = F.relu(new_query_min - tail - shift_of_node) + F.relu(tail + shift_of_node - new_query_max)
            score_center = new_query_center - tail - shift_of_node
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail + shift_of_node)) - new_query_center

        elif qtype == 'union-chain':
            # union after chain
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)

            heads = torch.chunk(head, 2, dim=0)

            query_center_1 = heads[0] + relations[0][:, 0, :, :]
            query_center_2 = heads[1] + relations[1][:, 0, :, :]
            query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:, 0, :, :])
            query_min_2 = query_center_2 - 0.5 * self.func(offsets[1][:, 0, :, :])
            query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:, 0, :, :])
            query_max_2 = query_center_2 + 0.5 * self.func(offsets[1][:, 0, :, :])
            query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1, query_max_1)
            query_center_2, query_min_2, query_max_2 = self.center_trans(query_center_2, query_min_2, query_max_2)

            query_center_1 = query_center_1 + relations[2][:, 0, :, :]
            query_center_2 = query_center_2 + relations[2][:, 0, :, :]
            query_min_1 = query_center_1 - 0.5 * self.func(offsets[2][:, 0, :, :])
            query_min_2 = query_center_2 - 0.5 * self.func(offsets[2][:, 0, :, :])
            query_max_1 = query_center_1 + 0.5 * self.func(offsets[2][:, 0, :, :])
            query_max_2 = query_center_2 + 0.5 * self.func(offsets[2][:, 0, :, :])
            query_center_1, query_min_1, query_max_1 = self.center_trans(query_center_1, query_min_1, query_max_1)
            query_center_2, query_min_2, query_max_2 = self.center_trans(query_center_2, query_min_2, query_max_2)



            new_query_min = torch.stack([query_min_1, query_min_2], dim=0)
            new_query_max = torch.stack([query_max_1, query_max_2], dim=0)
            new_query_center = torch.stack([query_center_1, query_center_2], dim=0)
            score_offset = F.relu(new_query_min - tail - shift_of_node) + F.relu(tail + shift_of_node - new_query_max)
            score_center = new_query_center - tail - shift_of_node
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail + shift_of_node)) - new_query_center

        else:
            query_center = head
            # consecutive add
            # need to calculate query_center, query_min, query_max
            # query_min, query_max are used to denote box
            for rel in range(rel_len):
                query_center = query_center + relation[:, rel, :, :]
                query_min = query_center - 0.5 * self.func(offset[:, rel, :, :])
                query_max = query_center + 0.5 * self.func(offset[:, rel, :, :])
                query_center, query_min, query_max = self.center_trans(query_center, query_min, query_max)

            if 'inter' not in qtype and 'union' not in qtype and 'disjoin' not in qtype:
                # all chain
                # execute here
                score_offset = F.relu(query_min - tail - shift_of_node) + F.relu(tail + shift_of_node - query_max)
                score_center = query_center - tail - shift_of_node
                score_center_plus = torch.min(query_max, torch.max(query_min, tail + shift_of_node)) - query_center
            else:
                rel_len = int(qtype.split('-')[0])
                assert rel_len > 1
                queries_min = torch.chunk(query_min, rel_len, dim=0)
                queries_max = torch.chunk(query_max, rel_len, dim=0)
                queries_center = torch.chunk(query_center, rel_len, dim=0)
                tails = torch.chunk(tail, rel_len, dim=0)
                offsets = query_max - query_min
                offsets = torch.chunk(offsets, rel_len, dim=0)

                # --- shift
                shift = torch.chunk(shift_of_node, rel_len, dim=0)

                if 'inter' in qtype:
                    if rel_len == 2:
                        new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                            queries_center[1].squeeze(1), offsets[1].squeeze(1))
                        new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                      queries_center[1].squeeze(1), offsets[1].squeeze(1))

                    elif rel_len == 3:
                        new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                            queries_center[1].squeeze(1), offsets[1].squeeze(1),
                                                            queries_center[2].squeeze(1), offsets[2].squeeze(1))
                        new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                      queries_center[1].squeeze(1), offsets[1].squeeze(1),
                                                      queries_center[2].squeeze(1), offsets[2].squeeze(1))
                    new_query_min = (new_query_center - 0.5 * self.func(new_offset)).unsqueeze(1)
                    new_query_max = (new_query_center + 0.5 * self.func(new_offset)).unsqueeze(1)
                    # if tail inside this box, the score_offset will be zero, otherwise bigger than 0
                    # this is the outside box distance

                    if rel_len == 2:
                        true_shift = shift[0] + shift[1]
                        true_shift = true_shift / 2
                    elif rel_len == 3:
                        true_shift = shift[0] + shift[1] + shift[2]
                        true_shift = true_shift / 3

                    score_offset = F.relu(new_query_min - tails[0] - true_shift) + F.relu(tails[0] + true_shift - new_query_max)
                    score_center = new_query_center.unsqueeze(1) - tails[0] - true_shift
                    # if tail inside the box, score_center_plus = tail
                    # otherwise, score_center_plus = new_query_max or new_query_min
                    # this is the inside box distance
                    score_center_plus = torch.min(new_query_max,
                                                  torch.max(new_query_min, tails[0] + true_shift)) - new_query_center.unsqueeze(1)
                elif 'union' in qtype:
                    new_query_min = torch.stack(queries_min, dim=0)
                    new_query_max = torch.stack(queries_max, dim=0)
                    new_query_center = torch.stack(queries_center, dim=0)
                    new_shift = torch.stack(shift, dim=0)
                    score_offset = F.relu(new_query_min - tails[0] - new_shift) + F.relu(tails[0] + new_shift - new_query_max)
                    score_center = new_query_center - tails[0] - new_shift
                    score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tails[0] + new_shift)) - new_query_center
                elif 'disjoin' in qtype:
                    if rel_len == 2:
                        new_query_center, new_query_min, new_query_max = self.disjoin_nn(queries_center[0], queries_min[0], queries_max[0],
                                                                                         queries_center[1], queries_min[1], queries_max[1])
                    elif rel_len == 3:
                        new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                            queries_center[1].squeeze(1), offsets[1].squeeze(1))
                        new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                      queries_center[1].squeeze(1), offsets[1].squeeze(1))

                        new_query_min = (new_query_center - 0.5 * self.func(new_offset)).unsqueeze(1)
                        new_query_max = (new_query_center + 0.5 * self.func(new_offset)).unsqueeze(1)

                        new_query_center, new_query_min, new_query_max = self.disjoin_nn(new_query_center.unsqueeze(1), new_query_min, new_query_max,
                                                                                         queries_center[2], queries_min[2], queries_max[2])

                    #if rel_len == 2:
                    #    true_shift = shift[0] + shift[1]
                    #    true_shift = true_shift / 2
                    #elif rel_len == 3:
                    #    true_shift = shift[0] + shift[1] + shift[2]
                    #    true_shift = true_shift / 3

                    #true_shift = shift[0]

                    score_offset = F.relu(new_query_min - tails[0]) + F.relu(tails[0] - new_query_max)
                    score_center = new_query_center - tails[0]
                    # if tail inside the box, score_center_plus = tail
                    # otherwise, score_center_plus = new_query_max or new_query_min
                    # this is the inside box distance
                    score_center_plus = torch.min(new_query_max,
                                                  torch.max(new_query_min, tails[0])) - new_query_center

                else:
                    assert False, 'qtype not exists: %s' % qtype
        score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)
        score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)
        score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(
            score_center_plus, p=1, dim=-1)
        if 'union' in qtype:
            score = torch.max(score, dim=0)[0]
            score_center = torch.max(score_center, dim=0)[0]
            score_center_plus = torch.max(score_center_plus, dim=0)[0]
        # for box embedding, only use socre_center_plus
        # for vec embedding, only use score
        # negative_score = F.logsigmoid(-negative_score_cen_plus).mean(dim=1)
        return score, score_center, torch.mean(torch.norm(offset, p=2, dim=2).squeeze(1)), score_center_plus, None

    def TransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype, shift_of_node=None):

        if qtype == 'chain-inter':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = (heads[0] + relations[0][:, 0, :, :] + relations[1][:, 0, :, :]).squeeze(1)
            score_2 = (heads[1] + relations[2][:, 0, :, :]).squeeze(1)
            conj_score = self.transE_deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score - tail
        elif qtype == 'inter-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = (heads[0] + relations[0][:, 0, :, :]).squeeze(1)
            score_2 = (heads[1] + relations[1][:, 0, :, :]).squeeze(1)
            conj_score = self.transE_deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score + relations[2][:, 0, :, :] - tail
        elif qtype == 'union-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = heads[0] + relations[0][:, 0, :, :] + relations[2][:, 0, :, :]
            score_2 = heads[1] + relations[1][:, 0, :, :] + relations[2][:, 0, :, :]
            conj_score = torch.stack([score_1, score_2], dim=0)
            score = conj_score - tail
        else:
            score = head
            for rel in range(rel_len):
                score = score + relation[:, rel, :, :]

            if 'inter' not in qtype and 'union' not in qtype:
                score = score - tail
            else:
                rel_len = int(qtype.split('-')[0])
                assert rel_len > 1
                score = score.squeeze(1)
                scores = torch.chunk(score, rel_len, dim=0)
                tails = torch.chunk(tail, rel_len, dim=0)
                if 'inter' in qtype:
                    if rel_len == 2:
                        conj_score = self.transE_deepsets(scores[0], None, scores[1], None)
                    elif rel_len == 3:
                        conj_score = self.transE_deepsets(scores[0], None, scores[1], None, scores[2], None)
                    conj_score = conj_score.unsqueeze(1)
                    score = conj_score - tails[0]
                elif 'union' in qtype:
                    conj_score = torch.stack(scores, dim=0)
                    score = conj_score - tails[0]
                else:
                    assert False, 'qtype not exist: %s' % qtype

        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        if 'union' in qtype:
            score = torch.max(score, dim=0)[0]
        if qtype == '2-union':
            score = score.unsqueeze(0)
        return score, None, None, 0., []

    @staticmethod
    # log = query2box.train_step(query2box, optimizer, train_iterator_2i, args, step)
    def train_step(model, optimizer, train_iterator, args, step):
        qtype1 = train_iterator.qtype
        #print(qtype1)
        model.train()
        optimizer.zero_grad()

        #start = time.time()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        #done = time.time()
        #elapsed = done - start
        #print("@@@ time: ", str(elapsed))

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        rel_len = int(train_iterator.qtype.split('-')[0])
        qtype = train_iterator.qtype
        negative_score, negative_score_cen, negative_offset, negative_score_cen_plus, _, _ = model(
            (positive_sample, negative_sample), rel_len, qtype, mode=mode)

        if model.geo == 'box':
            negative_score = F.logsigmoid(-negative_score_cen_plus).mean(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score, positive_score_cen, positive_offset, positive_score_cen_plus, _, _ = model(positive_sample,
                                                                                                   rel_len, qtype)
        if model.geo == 'box':
            positive_score = F.logsigmoid(positive_score_cen_plus).squeeze(dim=1)
        else:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

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
        if qtype == 'chain-inter' or qtype == 'inter-chain' or qtype == 'union-chain' or qtype == 'disjoin-chain':
            rel_len = 2
        else:
            rel_len = int(test_triples[0][-1].split('-')[0])

        model.eval()

        if qtype == 'inter-chain' or qtype == 'union-chain' or qtype == 'disjoin-chain':
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
        elif 'inter' in qtype or 'union' in qtype or 'disjoin' in qtype:
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

                    if 'inter' in qtype or 'disjoin' in qtype:
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

                    score -= (torch.min(score) - 1)
                    ans = test_ans[query]
                    hard_ans = test_ans_hard[query]
                    all_idx = set(range(args.nentity))
                    false_ans = all_idx - ans
                    ans_list = list(ans)
                    hard_ans_list = list(hard_ans)
                    if len(hard_ans) >= 1848:
                        bbb = 1
                        hard_ans_list = hard_ans_list[0:1500]

                    false_ans_list = list(false_ans)
                    ans_idxs = np.array(hard_ans_list)
                    vals = np.zeros((len(ans_idxs), args.nentity))
                    vals[np.arange(len(ans_idxs)), ans_idxs] = 1
                    axis2 = np.tile(false_ans_list, len(ans_idxs))
                    axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
                    vals[axis1, axis2] = 1
                    b = torch.Tensor(vals) if not args.cuda else torch.Tensor(vals).cuda()
                    filter_score = b * score

                    # my code
                    covid = False
                    if covid:
                        vals_ = np.ones((1, args.nentity))
                        b_ = torch.Tensor(vals_) if not args.cuda else torch.Tensor(vals_).cuda()
                        score_ = b_ * score
                        argsort_ = torch.argsort(score_, dim=1, descending=True)
                        res = argsort_.cpu().data.numpy()
                        print(res[0][0:10])
                    # ------

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

    @staticmethod
    def cal_distance(center_embed, box_min, box_max, candidates_embed, shift=None):
        # distance = center_embed - candidates_embed
        # score = -torch.norm(distance, p=1, dim=-1)
        if shift is None:
            score_offset = F.relu(box_min - candidates_embed) + F.relu(candidates_embed - box_max)
            score_center_plus = torch.min(box_max, torch.max(box_min, candidates_embed)) - center_embed
            score_center_plus = - torch.norm(score_offset, p=1, dim=-1) - 0.02 * torch.norm(score_center_plus, p=1,
                                                                                            dim=-1)
        else:
            score_offset = F.relu(box_min - candidates_embed - shift) + F.relu(candidates_embed + shift - box_max)
            score_center_plus = torch.min(box_max, torch.max(box_min, candidates_embed + shift)) - center_embed
            score_center_plus = - torch.norm(score_offset, p=1, dim=-1) - 0.02 * torch.norm(score_center_plus, p=1, dim=-1)

        return score_center_plus

    @staticmethod
    def sort_score_get_k(score, k):
        argsort_ = torch.argsort(score, dim=0, descending=True)
        return argsort_[0:k, ]

    @staticmethod
    def check_accuracy(model, variable, mapping, res_mapping):
        if variable == '5':
            _a = 1

        negative_sample = torch.LongTensor(range(model.nentity))
        all_embed = torch.index_select(model.entity_embedding, dim=0, index=negative_sample.cuda()).unsqueeze(1)
        res_center = res_mapping[variable]['center']
        res_min = res_mapping[variable]['query_min']
        res_max = res_mapping[variable]['query_max']

        shifts = res_mapping[variable]['in_predicates']
        all_shifts = torch.index_select(model.shift_embedding, dim=0, index=torch.LongTensor(shifts).cuda()).unsqueeze(1)
        all_shifts = torch.sum(all_shifts, dim=0)
        all_shifts = all_shifts / len(shifts)

        scores = Query2box.cal_distance(res_center, res_min, res_max, all_embed, all_shifts.unsqueeze(1))
        # just return the top 10 as answer
        res = Query2box.sort_score_get_k(scores, 10)
        res_np = res.cpu().numpy()
        answers = mapping[variable]
        index = 0
        res_set = set()
        for e in res_np:
            e = e[0]
            if e in answers:
                index = index + 1
                res_set.add(e)

        denominator = 10
        if denominator > len(answers):
            denominator = len(answers)
        return (index * 1.0 / denominator, res_set, denominator)

    @staticmethod
    def check_accuracy_with_union(model, union_variable, mapping, res_mapping):
        negative_sample = torch.LongTensor(range(model.nentity))
        all_embed = torch.index_select(model.entity_embedding, dim=0, index=negative_sample.cuda()).unsqueeze(1)

        res_center = res_mapping[union_variable]['center']
        res_min = res_mapping[union_variable]['query_min']
        res_max = res_mapping[union_variable]['query_max']

        new_query_min = torch.stack(res_min, dim=0)
        new_query_max = torch.stack(res_max, dim=0)
        new_query_center = torch.stack(res_center, dim=0)

        scores = Query2box.cal_distance(new_query_center, new_query_min, new_query_max, all_embed)
        scores = torch.max(scores, dim=0)[0]
        # just return the top 10 as answer
        res = Query2box.sort_score_get_k(scores, 10)
        res_np = res.cpu().numpy()
        answers = mapping[union_variable]
        index = 0
        res_set = set()
        for e in res_np:
            e = e[0]
            if e in answers:
                index = index + 1
                res_set.add(e)

        denominator = 10
        if denominator > len(answers):
            denominator = len(answers)
        return (index * 1.0 / denominator, res_set, denominator)

    @staticmethod
    def dig_subgraph(logical_triples, subgraph_info_map, from_to_map):
        # iterate each triple in the logical_triples
        for element in logical_triples:
            operation = element[0]
            if operation == 'anchor':
                all_anchors = element[1]
                for n in all_anchors:
                    xx_set = set()
                    xx_set.add(subgraph_info_map[n])
                    subgraph_info_map[n] = xx_set
            elif operation == 'projection':
                from_id = element[1]
                predicate_id = element[2]
                predicate_id = subgraph_info_map[predicate_id]
                result_id = element[3]
                if from_id not in subgraph_info_map:
                    raise ValueError('key %s not in result_map' % str(from_id))

                from_set = subgraph_info_map[from_id]
                to_set = subgraph_info_map[result_id]
                delete_set = set()
                for e in to_set:
                    is_good = False
                    for e2 in from_set:
                        if predicate_id not in from_to_map[e2]:
                            continue
                        if e in from_to_map[e2][predicate_id]:
                            is_good = True
                            break
                    if not is_good:
                        delete_set.add(e)

                subgraph_info_map[result_id] = to_set - delete_set
            elif operation == 'intersection':
                # only two intersection or three intersection
                data = element[1]
                result_id = element[2]
                if len(data) == 2:
                    head_id1 = data[0][0]
                    rel_id1 = data[0][1]
                    rel_id1 = subgraph_info_map[rel_id1]
                    head_id2 = data[1][0]
                    rel_id2 = data[1][1]
                    rel_id2 = subgraph_info_map[rel_id2]

                    from_set1 = subgraph_info_map[head_id1]
                    from_set2 = subgraph_info_map[head_id2]
                    to_set = subgraph_info_map[result_id]
                    delete_set = set()
                    for e in to_set:
                        good1 = False
                        good2 = False
                        for e2 in from_set1:
                            if good1:
                                break
                            if rel_id1 not in from_to_map[e2]:
                                continue
                            if e in from_to_map[e2][rel_id1]:
                                good1 = True
                                break
                        for e2 in from_set2:
                            if good2:
                                break
                            if rel_id2 not in from_to_map[e2]:
                                continue
                            if e in from_to_map[e2][rel_id2]:
                                good2 = True
                                break
                        if not good1 or not good2:
                            delete_set.add(e)
                    subgraph_info_map[result_id] = to_set - delete_set
                elif len(data) == 3:
                    head_id1 = data[0][0]
                    rel_id1 = data[0][1]
                    rel_id1 = subgraph_info_map[rel_id1]
                    head_id2 = data[1][0]
                    rel_id2 = data[1][1]
                    rel_id2 = subgraph_info_map[rel_id2]
                    head_id3 = data[2][0]
                    rel_id3 = data[2][1]
                    rel_id3 = subgraph_info_map[rel_id3]

                    from_set1 = subgraph_info_map[head_id1]
                    from_set2 = subgraph_info_map[head_id2]
                    from_set3 = subgraph_info_map[head_id3]
                    to_set = subgraph_info_map[result_id]
                    delete_set = set()
                    for e in to_set:
                        good1 = False
                        good2 = False
                        good3 = False
                        for e2 in from_set1:
                            if good1:
                                break
                            if rel_id1 not in from_to_map[e2]:
                                continue
                            if e in from_to_map[e2][rel_id1]:
                                good1 = True
                                break
                        for e2 in from_set2:
                            if good2:
                                break
                            if rel_id2 not in from_to_map[e2]:
                                continue
                            if e in from_to_map[e2][rel_id2]:
                                good2 = True
                                break
                        for e2 in from_set3:
                            if good3:
                                break
                            if rel_id3 not in from_to_map[e2]:
                                continue
                            if e in from_to_map[e2][rel_id3]:
                                good3 = True
                                break
                        if not good1 or not good2 or not good3:
                            delete_set.add(e)
                    subgraph_info_map[result_id] = to_set - delete_set

            elif operation == 'disjoin':
                head_id1 = element[1]
                rel_id1 = element[2]
                rel_id1 = subgraph_info_map[rel_id1]
                head_id2 = element[3]
                rel_id2 = element[4]
                rel_id2 = subgraph_info_map[rel_id2]
                result_id = element[5]

                from_set1 = subgraph_info_map[head_id1]
                from_set2 = subgraph_info_map[head_id2]
                to_set = subgraph_info_map[result_id]
                delete_set = set()
                for e in to_set:
                    good1 = False
                    good2 = True
                    for e2 in from_set1:
                        if good1:
                            break
                        if rel_id1 not in from_to_map[e2]:
                            continue
                        if e in from_to_map[e2][rel_id1]:
                            good1 = True
                            break
                    for e2 in from_set2:
                        if not good2:
                            break
                        if rel_id2 not in from_to_map[e2]:
                            continue
                        if e in from_to_map[e2][rel_id2]:
                            good2 = False
                            break
                    if not good1 or not good2:
                        delete_set.add(e)
                subgraph_info_map[result_id] = to_set - delete_set

            elif operation == 'union':
                head_id1 = element[1]
                rel_id1 = element[2]
                rel_id1 = subgraph_info_map[rel_id1]
                head_id2 = element[3]
                rel_id2 = element[4]
                rel_id2 = subgraph_info_map[rel_id2]
                result_id = element[5]

                from_set1 = subgraph_info_map[head_id1]
                from_set2 = subgraph_info_map[head_id2]
                to_set = subgraph_info_map[result_id]
                delete_set = set()
                for e in to_set:
                    good1 = False
                    good2 = False
                    for e2 in from_set1:
                        if good1:
                            break
                        if rel_id1 not in from_to_map[e2]:
                            continue
                        if e in from_to_map[e2][rel_id1]:
                            good1 = True
                            break
                    for e2 in from_set2:
                        if good2:
                            break
                        if rel_id2 not in from_to_map[e2]:
                            continue
                        if e in from_to_map[e2][rel_id2]:
                            good2 = True
                            break
                    if not good1 and not good2:
                        delete_set.add(e)
                subgraph_info_map[result_id] = to_set - delete_set

        return subgraph_info_map



    @staticmethod
    def cal_subgraph_acc(model, query_file, from_to_map):
        with open(query_file, 'rb') as handle:
            subgraph_5_star = pickle.load(handle)

        result_acc_map = {}
        subgraph_acc_map = {}
        for e in subgraph_5_star:
            query_triples = e[0]
            mapping = e[1]
            variables = e[2]
            res_map = model(find_subgraph=True, logical_triples=query_triples, subgraph_info_map=mapping)
            info_map2 = mapping.copy()
            for v in variables:
                acc, cands, denominator = Query2box.check_accuracy(model, v, mapping, res_map)
                if v not in result_acc_map:
                    result_acc_map[v] = []
                result_acc_map[v].append((acc, cands, denominator))
                info_map2[v] = cands

            xxx = Query2box.dig_subgraph(query_triples, info_map2, from_to_map)
            for v in variables:
                if v not in subgraph_acc_map:
                    subgraph_acc_map[v] = []
                subgraph_acc_map[v].append((len(xxx[v]), result_acc_map[v][-1][2]))


        for key in result_acc_map.keys():
            sum = 0
            acc_list = result_acc_map[key]
            for ele in acc_list:
                sum = sum + ele[0]
            average = sum * 1.0 / len(acc_list)
            result_acc_map[key] = average

        for key in subgraph_acc_map.keys():
            sum = 0
            acc_list = subgraph_acc_map[key]
            for ele in acc_list:
                sum = sum + ele[0] * 1.0 / ele[1]
            average = sum * 1.0 / len(acc_list)
            subgraph_acc_map[key] = average
        return result_acc_map, subgraph_acc_map

    @staticmethod
    def cal_subgraph_acc_union(model, query_file, from_to_map):
        with open(query_file, 'rb') as handle:
            subgraph_5_star = pickle.load(handle)

        result_acc_map = {}
        subgraph_acc_map = {}
        for e in subgraph_5_star:
            query_triples = e[0]
            mapping = e[1]
            variables = e[2]
            res_map = model(find_subgraph=True, logical_triples=query_triples, subgraph_info_map=mapping)
            info_map2 = mapping.copy()

            for v in variables[0:-1]:
                acc, cands, demoninator = Query2box.check_accuracy(model, v, mapping, res_map)
                if v not in result_acc_map:
                    result_acc_map[v] = []
                result_acc_map[v].append((acc, cands, demoninator))
                info_map2[v] = cands
            v = variables[-1]
            acc, cands, demoninator = Query2box.check_accuracy_with_union(model, v, mapping, res_map)
            if v not in result_acc_map:
                result_acc_map[v] = []
            result_acc_map[v].append((acc, cands, demoninator))
            info_map2[v] = cands


            xxx = Query2box.dig_subgraph(query_triples, info_map2, from_to_map)
            for v in variables:
                if v not in subgraph_acc_map:
                    subgraph_acc_map[v] = []
                subgraph_acc_map[v].append((len(xxx[v]), result_acc_map[v][-1][2]))

        for key in result_acc_map.keys():
            sum = 0
            acc_list = result_acc_map[key]
            for ele in acc_list:
                sum = sum + ele[0]
            average = sum * 1.0 / len(acc_list)
            result_acc_map[key] = average

        for key in subgraph_acc_map.keys():
            sum = 0
            acc_list = subgraph_acc_map[key]
            for ele in acc_list:
                sum = sum + ele[0] * 1.0 / ele[1]
            average = sum * 1.0 / len(acc_list)
            subgraph_acc_map[key] = average

        return result_acc_map, subgraph_acc_map

    @staticmethod
    def query_step(model):

        data_path = r"..\data\NELL"

        map_file = data_path + "/from_to_map.pkl"
        with open(map_file, 'rb') as handle:
            from_to_map = pickle.load(handle)

        query_file = data_path + "\subgraph_5_star.pkl"
        result_acc_map, sub = Query2box.cal_subgraph_acc(model, query_file, from_to_map)
        logging.info(result_acc_map)
        logging.info(sub)

        query_file = data_path + "\subgraph_6_disjoin.pkl"
        result_acc_map, sub = Query2box.cal_subgraph_acc(model, query_file, from_to_map)
        logging.info(result_acc_map)
        logging.info(sub)

        query_file = data_path + "\subgraph_6_star.pkl"
        result_acc_map, sub = Query2box.cal_subgraph_acc(model, query_file, from_to_map)
        logging.info(result_acc_map)
        logging.info(sub)

        query_file = data_path + "\subgraph_7_disjoin.pkl"
        result_acc_map, sub = Query2box.cal_subgraph_acc(model, query_file, from_to_map)
        logging.info(result_acc_map)
        logging.info(sub)

        query_file = data_path + "\subgraph_6_union.pkl"
        result_acc_map, sub = Query2box.cal_subgraph_acc_union(model, query_file, from_to_map)
        logging.info(result_acc_map)
        logging.info(sub)

        query_file = data_path + "\subgraph_7_union.pkl"
        result_acc_map, sub = Query2box.cal_subgraph_acc_union(model, query_file, from_to_map)
        logging.info(result_acc_map)
        logging.info(sub)

