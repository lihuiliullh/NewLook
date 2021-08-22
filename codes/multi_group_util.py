from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F


def prepare_data(sample, qtype, mode, group_times, node_group_one_hot_vector_multi, group_adj_matrix_multi, rel_len=None):

    if qtype == 'chain-inter':
        assert mode == 'tail-batch'
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

        # -------------------------group
        # -------------------group
        one_hot_head = []
        tail_one_hot = []
        for group_ts in range(group_times):
            one_hot_head_1 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                index=head_part[:, 0]).unsqueeze(
                1)
            one_hot_head_2 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                index=head_part[:, 3]).unsqueeze(
                1)
            one_hot_head__ = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)

            tail_one_hot__ = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                index=tail_part.view(-1)).view(batch_size,
                                                                               negative_sample_size,
                                                                               -1)
            one_hot_head.append(one_hot_head__)
            tail_one_hot.append(tail_one_hot__)
        # -------------------group
        # -----------------------relation matrix
        relation_matrix = []
        for group_ts in range(group_times):
            relation_matrix_11 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                    index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_matrix_12 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                    index=head_part[:, 2]).unsqueeze(
                1).unsqueeze(1)
            relation_matrix_2 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                   index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation_matrix__ = torch.cat([relation_matrix_11, relation_matrix_12, relation_matrix_2], dim=0)
            relation_matrix.append(relation_matrix__)
        # -----------------------relation matrix
        # -------------------------group

    elif qtype == 'inter-chain' or qtype == 'union-chain' or qtype == 'disjoin-chain':
        assert mode == 'tail-batch'
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

        # -------------------------group
        # -------------------group
        one_hot_head = []
        tail_one_hot = []
        for group_ts in range(group_times):
            one_hot_head_1 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                index=head_part[:, 0]).unsqueeze(
                1)
            one_hot_head_2 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                index=head_part[:, 2]).unsqueeze(
                1)
            one_hot_head__ = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)

            tail_one_hot__ = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                index=tail_part.view(-1)).view(batch_size,
                                                                               negative_sample_size,
                                                                               -1)
            one_hot_head.append(one_hot_head__)
            tail_one_hot.append(tail_one_hot__)
        # -------------------group
        # -----------------------relation matrix
        relation_matrix = []
        for group_ts in range(group_times):
            relation_matrix_11 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                    index=head_part[:, 1]).unsqueeze(
                1).unsqueeze(1)
            relation_matrix_12 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                    index=head_part[:, 3]).unsqueeze(
                1).unsqueeze(1)
            relation_matrix_2 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                   index=head_part[:, 4]).unsqueeze(
                1).unsqueeze(1)
            relation_matrix__ = torch.cat([relation_matrix_11, relation_matrix_12, relation_matrix_2], dim=0)
            relation_matrix.append(relation_matrix__)
        # -----------------------relation matrix
        # -------------------------group

    elif qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union' or qtype == '2-disjoin' or qtype == '3-disjoin':
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            # -------------------group
            one_hot_head = []
            tail_one_hot = []
            for group_ts in range(group_times):
                one_hot_head_1 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=sample[:, 0]).unsqueeze(
                    1)
                one_hot_head_2 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=sample[:, 2]).unsqueeze(
                    1)
                one_hot_head__ = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)
                if rel_len == 3:
                    one_hot_head_3 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                        index=sample[:, 4]).unsqueeze(1)
                    one_hot_head__ = torch.cat([one_hot_head__, one_hot_head_3], dim=0)

                tail_one_hot__ = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=sample[:, -1]).unsqueeze(1)
                if rel_len == 2:
                    tail_one_hot__ = torch.cat([tail_one_hot__, tail_one_hot__], dim=0)
                elif rel_len == 3:
                    tail_one_hot__ = torch.cat([tail_one_hot__, tail_one_hot__, tail_one_hot__], dim=0)

                one_hot_head.append(one_hot_head__)
                tail_one_hot.append(tail_one_hot__)
            # -------------------group
            # -----------------------relation matrix
            relation_matrix = []
            for group_ts in range(group_times):
                relation_matrix_1 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                       index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix_2 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                       index=sample[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix__ = torch.cat([relation_matrix_1, relation_matrix_2], dim=0)
                if rel_len == 3:
                    relation_matrix_3 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                           index=sample[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix__ = torch.cat([relation_matrix__, relation_matrix_3], dim=0)

                relation_matrix.append(relation_matrix__)
            # -----------------------relation matrix

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            # -------------------group
            one_hot_head = []
            tail_one_hot = []
            for group_ts in range(group_times):
                one_hot_head_1 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=head_part[:, 0]).unsqueeze(
                    1)
                one_hot_head_2 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=head_part[:, 2]).unsqueeze(
                    1)
                one_hot_head__ = torch.cat([one_hot_head_1, one_hot_head_2], dim=0)
                if rel_len == 3:
                    one_hot_head_3 = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                        index=head_part[:, 4]).unsqueeze(1)
                    one_hot_head__ = torch.cat([one_hot_head__, one_hot_head_3], dim=0)

                tail_one_hot__ = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                if rel_len == 2:
                    tail_one_hot__ = torch.cat([tail_one_hot__, tail_one_hot__], dim=0)
                elif rel_len == 3:
                    tail_one_hot__ = torch.cat([tail_one_hot__, tail_one_hot__, tail_one_hot__], dim=0)

                one_hot_head.append(one_hot_head__)
                tail_one_hot.append(tail_one_hot__)
            # -------------------group
            # -----------------------relation matrix
            relation_matrix = []
            for group_ts in range(group_times):
                relation_matrix_1 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                       index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix_2 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                       index=head_part[:, 3]).unsqueeze(
                    1).unsqueeze(1)
                relation_matrix__ = torch.cat([relation_matrix_1, relation_matrix_2], dim=0)
                if rel_len == 3:
                    relation_matrix_3 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                           index=head_part[:, 5]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix__ = torch.cat([relation_matrix__, relation_matrix_3], dim=0)

                relation_matrix.append(relation_matrix__)
            # -----------------------relation matrix

    elif qtype == '1-chain' or qtype == '2-chain' or qtype == '3-chain':
        if mode == 'single':
            # just one negative or positive sample
            batch_size, negative_sample_size = sample.size(0), 1

            # -------------------group
            one_hot_head = []
            tail_one_hot = []
            for group_ts in range(group_times):
                one_hot_head__ = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=sample[:, 0]).unsqueeze(1)

                tail_one_hot__ = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=sample[:, -1]).unsqueeze(1)
                one_hot_head.append(one_hot_head__)
                tail_one_hot.append(tail_one_hot__)
            # -------------------group
            # -----------------------relation matrix
            relation_matrix = []
            for group_ts in range(group_times):
                relation_matrix__ = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                       index=sample[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                if rel_len == 2 or rel_len == 3:
                    relation_matrix_2 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                           index=sample[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix__ = torch.cat([relation_matrix__, relation_matrix_2], 1)
                if rel_len == 3:
                    relation_matrix_3 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                           index=sample[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix__ = torch.cat([relation_matrix__, relation_matrix_3], 1)

                relation_matrix.append(relation_matrix__)
            # -----------------------relation matrix

        elif mode == 'tail-batch':
            # batch size
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            # -------------------group
            one_hot_head = []
            tail_one_hot = []
            for group_ts in range(group_times):
                one_hot_head__ = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=head_part[:, 0]).unsqueeze(1)

                tail_one_hot__ = torch.index_select(node_group_one_hot_vector_multi[group_ts], dim=0,
                                                    index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                one_hot_head.append(one_hot_head__)
                tail_one_hot.append(tail_one_hot__)
            # -------------------group
            # -----------------------relation matrix
            relation_matrix = []
            for group_ts in range(group_times):
                relation_matrix__ = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                       index=head_part[:, 1]).unsqueeze(
                    1).unsqueeze(1)

                if rel_len == 2 or rel_len == 3:
                    relation_matrix_2 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                           index=head_part[:, 2]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix__ = torch.cat([relation_matrix__, relation_matrix_2], 1)
                if rel_len == 3:
                    relation_matrix_3 = torch.index_select(group_adj_matrix_multi[group_ts], dim=0,
                                                           index=head_part[:, 3]).unsqueeze(
                        1).unsqueeze(1)
                    relation_matrix__ = torch.cat([relation_matrix__, relation_matrix_3], 1)

                relation_matrix.append(relation_matrix__)
            # -----------------------relation matrix

    return one_hot_head, relation_matrix, tail_one_hot

def run_multi_group(rel_len, qtype, group_times, head_one_hot, relation_matrix, tail_one_hot,
                    disjoin_weight_for_group_matrix, disjoin_then_proj_threshold):
    ##### group
    for group_ts in range(group_times):
        head_one_hot[group_ts] = torch.unsqueeze(head_one_hot[group_ts], 2)
        tail_one_hot[group_ts] = torch.unsqueeze(tail_one_hot[group_ts], 2)
    ############3 group
    if qtype == 'chain-inter':
        # ------------------group
        group_dist_list = []
        for group_ts in range(group_times):
            query_one_hots = torch.chunk(head_one_hot[group_ts], 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix[group_ts], 3, dim=0)
            # chain 1
            query_one_hot1 = query_one_hots[0]
            for i in range(2):
                query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[i][:, 0, :, :])
            # chain 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[2][:, 0, :, :])

            query_one_hot1[query_one_hot1 >= 1] = 1
            query_one_hot2[query_one_hot2 >= 1] = 1
            query_one_hot_res = query_one_hot1 + query_one_hot2
            query_one_hot_res[query_one_hot_res < 2] = 0
            query_one_hot_res[query_one_hot_res >= 2] = 1

            group_dist_ = F.relu(tail_one_hot[group_ts] - query_one_hot_res)
            group_dist_ = torch.squeeze(group_dist_, 2)
            group_dist_ = torch.norm(group_dist_, p=1, dim=2)
            group_dist_list.append(group_dist_)
        # ------------------group

    elif qtype == 'inter-chain' or qtype == 'disjoin-chain':
        # ------------------group
        group_dist_list = []
        for group_ts in range(group_times):
            query_one_hots = torch.chunk(head_one_hot[group_ts], 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix[group_ts], 3, dim=0)
            # pro 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            # pro 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[1][:, 0, :, :])

            # inter or disjoin
            if qtype == 'inter-chain':
                new_query_one_hot = query_one_hot1 + query_one_hot2
                new_query_one_hot[new_query_one_hot < 2] = 0
                new_query_one_hot[new_query_one_hot >= 2] = 1
            elif qtype == 'disjoin-chain':
                new_query_one_hot = query_one_hot1 - query_one_hot2
                new_query_one_hot[new_query_one_hot < 0] = 0

            new_query_one_hot = torch.matmul(new_query_one_hot, relation_matrix_chunk[2][:, 0, :, :])
            if qtype == 'disjoin-chain':
                new_query_one_hot[new_query_one_hot >= 1] = 1
                new_query_one_hot[new_query_one_hot < 1] = 0
            else:
                new_query_one_hot[new_query_one_hot >= disjoin_then_proj_threshold] = 1
                new_query_one_hot[new_query_one_hot < disjoin_then_proj_threshold] = 0

            group_dist_ = F.relu(tail_one_hot[group_ts] - new_query_one_hot)
            group_dist_ = torch.squeeze(group_dist_, 2)
            group_dist_ = torch.norm(group_dist_, p=1, dim=2)
            group_dist_list.append(group_dist_)
        # ------------------group

        #######################################################################################

    elif qtype == 'union-chain':
        # ------------------group
        group_dist_list = []
        for group_ts in range(group_times):
            query_one_hots = torch.chunk(head_one_hot[group_ts], 2, dim=0)
            relation_matrix_chunk = torch.chunk(relation_matrix[group_ts], 3, dim=0)
            # pro 1
            query_one_hot1 = query_one_hots[0]
            query_one_hot1 = torch.matmul(query_one_hot1, relation_matrix_chunk[0][:, 0, :, :])
            # pro 2
            query_one_hot2 = query_one_hots[1]
            query_one_hot2 = torch.matmul(query_one_hot2, relation_matrix_chunk[1][:, 0, :, :])

            new_query_one_hot = query_one_hot1 + query_one_hot2
            new_query_one_hot[new_query_one_hot < 1] = 0
            new_query_one_hot[new_query_one_hot >= 1] = 1

            new_query_one_hot = torch.matmul(query_one_hot2, relation_matrix_chunk[2][:, 0, :, :])
            new_query_one_hot[new_query_one_hot >= 1] = 1
            new_query_one_hot[new_query_one_hot < 1] = 0

            group_dist_ = F.relu(tail_one_hot[group_ts] - new_query_one_hot)
            group_dist_ = torch.squeeze(group_dist_, 2)
            group_dist_ = torch.norm(group_dist_, p=1, dim=2)
            group_dist_list.append(group_dist_)
        # ------------------group

        #######################################################################################

    else:
        # ------------------group
        group_dist_list = []
        for group_ts in range(group_times):
            query_one_hot = head_one_hot[group_ts]
            for rel in range(rel_len):
                rel_m = relation_matrix[group_ts][:, rel, :, :]
                query_one_hot = torch.matmul(query_one_hot, rel_m)
            query_one_hot[query_one_hot >= 1] = 1
            if 'inter' not in qtype and 'union' not in qtype and 'disjoin' not in qtype:
                group_dist_ = F.relu(tail_one_hot[group_ts] - query_one_hot)
                group_dist_ = torch.squeeze(group_dist_, 2)
                group_dist_ = torch.norm(group_dist_, p=1, dim=2)
            else:
                rel_len_ = int(qtype.split('-')[0])
                new_query_one_hot_chunk = torch.chunk(query_one_hot, rel_len_, dim=0)
                tail_one_hot_chunk = torch.chunk(tail_one_hot[group_ts], rel_len_, dim=0)
                if 'inter' in qtype:
                    new_query_one_hot_ = new_query_one_hot_chunk[0]
                    for i in range(1, rel_len_, 1):
                        new_query_one_hot_ = new_query_one_hot_ + new_query_one_hot_chunk[i]

                    new_query_one_hot_[new_query_one_hot_ < rel_len_] = 0
                    new_query_one_hot_[new_query_one_hot_ >= 1] = 1
                    group_dist_ = F.relu(tail_one_hot_chunk[0] - new_query_one_hot_)
                    group_dist_ = torch.squeeze(group_dist_, 2)
                    group_dist_ = torch.norm(group_dist_, p=1, dim=2)
                elif 'union' in qtype:
                    new_query_one_hot_ = new_query_one_hot_chunk[0]
                    for i in range(1, rel_len_, 1):
                        new_query_one_hot_ = new_query_one_hot_ + new_query_one_hot_chunk[i]

                    new_query_one_hot_[new_query_one_hot_ < 1] = 0
                    new_query_one_hot_[new_query_one_hot_ >= 1] = 1
                    group_dist_ = F.relu(tail_one_hot_chunk[0] - new_query_one_hot_)
                    group_dist_ = torch.squeeze(group_dist_, 2)
                    group_dist_ = torch.norm(group_dist_, p=1, dim=2)
                elif 'disjoin' in qtype:
                    if rel_len_ == 2:
                        new_query_one_hot_ = new_query_one_hot_chunk[0] - disjoin_weight_for_group_matrix * \
                                             new_query_one_hot_chunk[1]
                        new_query_one_hot_[new_query_one_hot_ < 0] = 0
                        group_dist_ = F.relu(tail_one_hot_chunk[0] - new_query_one_hot_)
                        group_dist_ = torch.squeeze(group_dist_, 2)
                        group_dist_ = torch.norm(group_dist_, p=1, dim=2)
                    elif rel_len_ == 3:
                        new_query_one_hot_ = new_query_one_hot_chunk[0] - disjoin_weight_for_group_matrix * \
                                             new_query_one_hot_chunk[1]
                        new_query_one_hot_[new_query_one_hot_ < 0] = 0
                        new_query_one_hot_[new_query_one_hot_ > 0] = 1
                        new_query_one_hot_ = new_query_one_hot_ - disjoin_weight_for_group_matrix * \
                                             new_query_one_hot_chunk[2]
                        new_query_one_hot_[new_query_one_hot_ < 0] = 0
                        group_dist_ = F.relu(tail_one_hot_chunk[0] - new_query_one_hot_)
                        group_dist_ = torch.squeeze(group_dist_, 2)
                        group_dist_ = torch.norm(group_dist_, p=1, dim=2)

            group_dist_list.append(group_dist_)
        # ------------------group

        #####################################################################################

    group_dist_list = torch.stack(group_dist_list, dim=0)
    group_dist = torch.max(group_dist_list, dim=0)[0]
    return group_dist
