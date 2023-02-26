from distutils.version import LooseVersion
import logging
import math
import random
import six
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from argparse import Namespace

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy

from espnet.nets.pytorch_backend.nets_utils import mask_by_length
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.scorer_interface import ScorerInterface
from espnet.nets.pytorch_backend.lm.model_debug import RNNModel
# from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from espnet.nets.pytorch_backend.GAT import GAT

MAX_DECODER_OUTPUT = 5
CTC_SCORING_RATIO = 1.5


class Decoder(torch.nn.Module, ScorerInterface):
    """Decoder module

    :param int eprojs: encoder projection units
    :param int odim: dimension of outputs
    :param str dtype: gru or lstm
    :param int dlayers: decoder layers
    :param int dunits: decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    :param float sampling_probability: scheduled sampling probability
    :param float dropout: dropout rate
    :param float context_residual: if True, use context vector for token generation
    :param float replace_sos: use for multilingual (speech/text) translation
    """

    def __init__(
        self,
        eprojs,
        odim,
        dtype,
        dlayers,
        dunits,
        sos,
        eos,
        att,
        verbose=0,
        char_list=None,
        labeldist=None,
        lsm_weight=0.0,
        sampling_probability=0.0,
        dropout=0.0,
        context_residual=False,
        replace_sos=False,
        num_encs=1,
        wordemb=0, lm_odim=1, meetingKB=None, KBlextree=False, PtrGen=False,
        PtrSche=0, PtrKBin=False, smoothprob=1.0, attn_dim=0, acousticonly=True,
        additive=False, ooKBemb=False, DBmask=0.0, DBinput=False, treetype='',
        tree_hid=0, boostfactor=0, stacked=False
    ):

        torch.nn.Module.__init__(self)
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.context_residual = context_residual
        self.embed = torch.nn.Embedding(odim, dunits)
        self.dropout_emb = torch.nn.Dropout(p=dropout)
        self.dropout_KB = torch.nn.Dropout(p=dropout)

        # gs534 - KB related
        self.meetingKB = meetingKB
        self.bpeunk = char_list.index('<unk>') if '<unk>' in char_list else -1
        self.wordemb = wordemb
        self.ac_only = acousticonly
        self.useKBinput = False
        self.attn_dim = attn_dim if attn_dim != 0 else self.dunits
        self.additive = additive
        # self.ooKBemb = torch.nn.Embedding(1, dunits)
        embdim = 0
        self.useKBinput = (meetingKB is not None) and (not PtrGen or PtrKBin)
        if meetingKB is not None:
            # Define graph neural net parameters
            self.tree_hid = self.dunits if tree_hid == 0 else tree_hid
            if treetype == 'lstm':
                self.input_gate = torch.nn.Linear(self.dunits+self.tree_hid, self.tree_hid)
                self.forget_gate = torch.nn.Linear(self.dunits+self.tree_hid, self.tree_hid)
                self.output_gate = torch.nn.Linear(self.dunits+self.tree_hid, self.tree_hid)
                self.transform_gate = torch.nn.Linear(self.dunits+self.tree_hid, self.tree_hid)
            elif treetype.startswith('gcn'):
                self.gcn_l1 = torch.nn.Linear(self.dunits, self.tree_hid)
                self.gcn_l2 = torch.nn.Linear(self.tree_hid, self.tree_hid)
            elif treetype.startswith('gat'):
                self.nheads = [int(head) for head in treetype.split('_')[1:]]
                self.GAT = GAT(num_of_layers=len(self.nheads), num_heads_per_layer=self.nheads,
                               num_features_per_layer=[self.dunits]+[self.tree_hid]*len(self.nheads),
                               boost_self=boostfactor)
            elif treetype.startswith('sage'):
                self.sage_pool_1 = torch.nn.Linear(self.dunits, self.tree_hid)
                self.sage_merge_1 = torch.nn.Linear(self.dunits + self.tree_hid, self.tree_hid)
                self.sage_pool_2 = torch.nn.Linear(self.tree_hid, self.tree_hid)
                self.sage_merge_2 = torch.nn.Linear(self.tree_hid + self.tree_hid, self.tree_hid)
                self.sage_pool_3 = torch.nn.Linear(self.tree_hid, self.tree_hid)
                self.sage_merge_3 = torch.nn.Linear(self.tree_hid + self.tree_hid, self.tree_hid)
                self.sage_pool_4 = torch.nn.Linear(self.tree_hid, self.tree_hid)
                self.sage_merge_4 = torch.nn.Linear(self.tree_hid + self.tree_hid, self.tree_hid)
            else:
                self.tree_hid = self.dunits
                self.recursive_proj = torch.nn.Linear(self.dunits+self.tree_hid, self.tree_hid)
            # Define TCPGen parameters
            self.Qproj = torch.nn.Linear(self.dunits+eprojs, self.attn_dim)
            self.Kproj = torch.nn.Linear(self.tree_hid, self.attn_dim)
            self.Qprojclass = torch.nn.Linear(self.attn_dim, self.attn_dim)
            self.Kprojclass = torch.nn.Linear(self.attn_dim, self.attn_dim)
            if self.additive:
                self.AttnProj_1 = torch.nn.Linear(self.attn_dim*2, self.attn_dim*2)
                self.AttnProj_2 = torch.nn.Linear(self.attn_dim*2, 1)
            self.pointer_gate = torch.nn.Linear(self.attn_dim+self.dunits if self.ac_only else self.attn_dim*2, 1)
            self.ooKBemb = torch.nn.Embedding(1, self.tree_hid)
            embdim = self.attn_dim
        self.KBlextree = KBlextree
        self.PtrGen = PtrGen
        self.epoch = 0
        self.PtrSche = PtrSche
        self.smoothprob = smoothprob
        self.DBmask = DBmask
        self.treetype = treetype
        self.stacked = stacked
        # Using deep biasing
        self.DBinput = DBinput
        if meetingKB is not None and DBinput:
            self.DBembed = torch.nn.Linear(odim, embdim, bias=False)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(dunits + eprojs, dunits)
            if self.dtype == "lstm"
            else torch.nn.GRUCell(dunits + eprojs, dunits)
        ]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in six.moves.range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(dunits, dunits)
                if self.dtype == "lstm"
                else torch.nn.GRUCell(dunits, dunits)
            ]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf
        self.ignore_id = -1

        if self.useKBinput or self.DBinput:
            self.post_LSTM_proj = torch.nn.Linear(dunits+embdim, dunits)

        if context_residual:
            self.output = torch.nn.Linear(dunits + eprojs, odim)
        else:
            self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.spaceids = char_list.index('<space>') if '<space>' in char_list else -1
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs

        # for multilingual E2E-ST
        self.replace_sos = replace_sos

        self.logzero = -10000000000.0

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for i in six.moves.range(1, self.dlayers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), (z_prev[i], c_prev[i])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for i in six.moves.range(1, self.dlayers):
                z_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i]
                )
        return z_list, c_list

    def get_meetingKB_emb_map(self, query, meeting_mask, meeting_embs, back_transform):
        meeting_KB = self.dropout_KB(self.Kproj(meeting_embs))
        KBweight = torch.einsum('ijk,ik->ij', meeting_KB, query)
        KBweight = KBweight / math.sqrt(query.size(-1))
        KBweight.masked_fill_(meeting_mask.bool(), -1e9)
        KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
        if meeting_embs.size(1) > 1:
            KBembedding = torch.einsum('ijk,ij->ik', meeting_KB[:,:-1,:], KBweight[:,:-1])
        else:
            KBembedding = KBweight.new_zeros(meeting_KB.size(0), meeting_KB.size(-1))
        KBweight = torch.einsum('ijk,ij->ik', back_transform, KBweight)
        return KBembedding, KBweight

    def get_meetingKB_emb_map_forest(self, query, meeting_mask, meeting_embs, back_transform, stacked=False):
        meeting_KB = self.dropout_KB(self.Kproj(meeting_embs))
        KBweight = torch.einsum('ijk,k->ij', meeting_KB, query.squeeze(0))
        KBweight = KBweight / math.sqrt(query.size(-1))
        KBweight.masked_fill_(meeting_mask.bool(), -1e9)
        # option: only have one ooKB logit, and mask other logits
        if stacked:
            KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
            KBclassemb = torch.einsum('ijk,ij->ik', meeting_KB[:,:-1,:], KBweight[:,:-1])
            KBclassemb = self.dropout_KB(self.Kprojclass(KBclassemb))
            KBclassquery = self.dropout_KB(self.Qprojclass(query))
            KBweightclass = torch.einsum('ik,k->i', KBclassemb, KBclassquery) / math.sqrt(KBclassquery.size(-1))
            KBweightclass = torch.softmax(KBweightclass, dim=-1)
            KBembedding = torch.einsum('i,ik->k', KBweightclass, KBclassemb).unsqueeze(0)
            classpost = KBweightclass
            KBweight = KBweightclass.unsqueeze(1) * KBweight
        else:
            KBweight[1:,-1] = -1e9
            KBweight = KBweight.view(1, -1)
            KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
            KBweight = KBweight.view(meeting_KB.size(0), -1)
            if meeting_embs.size(1) > 1:
                KBembedding = torch.einsum('ijk,ij->ik', meeting_KB[:,:-1,:], KBweight[:,:-1])
                KBembedding = KBembedding.sum(dim=0, keepdim=True)
            else:
                KBembedding = KBweight.new_zeros(1, meeting_KB.size(-1))
            classpost = KBweight[:,:-1].sum(dim=1)
        KBweight = torch.einsum('ijk,ij->ik', back_transform, KBweight).sum(dim=0, keepdim=True)
        return KBembedding, KBweight, classpost

    def get_lextree_step_embs_inference(self, char_idx, tree_track, reset_tree):
        # meeting_KB = torch.cat([self.embed.weight.data, self.ooKBemb.weight], dim=0)
        ooKB_id = len(self.char_list)
        new_tree = tree_track[0]
        char_idx = char_idx if isinstance(char_idx, int) else char_idx.item()
        ptr_gen = True
        if char_idx in [self.eos, self.spaceids] or self.char_list[char_idx].endswith('▁'):
            new_tree = reset_tree # self.meetingKB.meetinglextree[meeting] if isinstance(meeting, str) else meeting
        elif char_idx not in new_tree:
            new_tree = [{}]
            ptr_gen = False
        else:
            new_tree = new_tree[char_idx]
        indices = list(new_tree[0].keys())
        if len(new_tree) > 2 and new_tree[0] != {}:
            step_embs = torch.cat([new_tree[0][key][3] for key in indices], dim=0)
        else:
            step_embs = torch.empty(0, self.tree_hid)
        back_transform = []
        indices += [ooKB_id] # list(new_tree[0].keys()) + [ooKB_id]
        for i, ind in enumerate(indices):
            one_hot = [0] * (ooKB_id + 1)
            one_hot[ind] = 1
            back_transform.append(one_hot)
        back_transform = torch.Tensor(back_transform)
        # step_embs = torch.einsum('jk,km->jm', back_transform, meeting_KB)
        step_embs = torch.cat([step_embs, self.ooKBemb.weight], dim=0)
        step_mask = torch.zeros(back_transform.size(0)).byte()
        return step_mask.unsqueeze(0), new_tree, ptr_gen, step_embs.unsqueeze(0), back_transform.unsqueeze(0)

    def get_lextree_step_embs(self, char_ids, trees, origTries, clm=False):
        # step_mask = char_ids.new_ones(char_ids.size(0), len(self.char_list)+1)
        # meeting_KB = torch.cat([self.embed.weight.data, self.ooKBemb.weight], dim=0)
        ooKB_id = len(self.char_list)
        maxlen = 0
        index_list = []
        new_trees = []
        p_gen_mask = []
        step_embs = []
        for i, vy in enumerate(char_ids):
            new_tree = trees[i][0]
            vy = vy.item()
            if vy in [self.eos] or (self.char_list[vy].endswith('▁') and not clm):
                new_tree = origTries[i]
                p_gen_mask.append(0)
            elif vy not in new_tree:
                new_tree = [{}]
                p_gen_mask.append(1)
            else:
                new_tree = new_tree[vy]
                if clm and self.char_list[vy].endswith('▁') and new_tree[0] == {}:
                    p_gen_mask.append(1)
                else:
                    p_gen_mask.append(0)
            new_trees.append(new_tree)
            if len(new_tree[0].keys()) > maxlen:
                maxlen = len(new_tree[0].keys())
            index_list.append(list(new_tree[0].keys()))
            if len(new_tree) > 2 and new_tree[0] != {}:
                # step_embs.append(torch.cat([node[3] for key, node in new_tree[0].items()], dim=0))
                step_embs.append(torch.cat([new_tree[0][key][3] for key in index_list[-1]]))
            else:
                step_embs.append(to_device(self, torch.empty(0, self.tree_hid)))

        # reset for clm entities
        if clm and self.char_list[char_ids[0]].endswith('▁') and 0 not in p_gen_mask:
            new_trees = origTries
            maxlen = 0
            index_list = []
            step_embs = []
            for new_tree in new_trees:
                if len(new_tree[0].keys()) > maxlen:
                    maxlen = len(new_tree[0].keys())
                index_list.append(list(new_tree[0].keys()))
                if len(new_tree) > 2 and new_tree[0] != {}:
                    step_embs.append(torch.cat([new_tree[0][key][3] for key in index_list[-1]]))
                else:
                    step_embs.append(to_device(self, torch.empty(0, self.tree_hid)))
            p_gen_mask = [0] * len(p_gen_mask)

        maxlen += 1
        step_mask = []
        back_transform = to_device(self, torch.zeros(char_ids.size(0), maxlen, ooKB_id+1))
        ones_mat = to_device(self, torch.ones(back_transform.size()))
        for i, indices in enumerate(index_list):
            step_mask.append(len(indices) * [0] + (maxlen - len(indices) - 1) * [1] + [0])
            pad_embs = self.ooKBemb.weight.repeat(maxlen-len(indices), 1)
            indices += [ooKB_id] * (maxlen-len(indices))
            step_embs[i] = torch.cat([step_embs[i], pad_embs], dim=0)
        step_mask = to_device(self, torch.tensor(step_mask).byte())
        index_list = to_device(self, torch.LongTensor(index_list))
        back_transform.scatter_(dim=-1, index=index_list.unsqueeze(-1), src=ones_mat)
        step_embs = torch.stack(step_embs)
        # step_embs = torch.einsum('ijk,km->ijm', back_transform, meeting_KB) # torch.stack(step_embs)
        return step_mask, new_trees, p_gen_mask, step_embs, back_transform

    def forward_treelstm_cell(self, hs, cs, xj):
        """Input arguments
        hs: (n_nodes, tree_hid)
        cs: (n_nodes, tree_hid)
        xj: (1, dunits)
        """
        h_j = hs.sum(dim=0).unsqueeze(0)
        xh_j = torch.cat([h_j, xj], dim=-1)
        i_j = torch.sigmoid(self.input_gate(xh_j))
        o_j = torch.sigmoid(self.output_gate(xh_j))
        u_j = torch.tanh(self.transform_gate(xh_j))

        xh_kj = torch.cat([hs, xj.repeat(hs.size(0), 1)], dim=-1)
        # forget gate (n_nodes, tree_hid)
        f_jk = torch.sigmoid(self.forget_gate(xh_kj))

        c_j = i_j * u_j + (f_jk * cs).sum(dim=0).unsqueeze(0)
        h_j = o_j * torch.tanh(c_j)
        return h_j, c_j

    def get_lextree_encs_treelstm(self, lextree, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            ey = self.embed(to_device(self, torch.LongTensor([wordpiece])))
            ey = self.dropout_KB(ey)
            hs, cs = ey.new_zeros(1, self.tree_hid), ey.new_zeros(1, self.tree_hid)
            hj, cj = self.forward_treelstm_cell(hs, cs, ey)
            lextree.append(hj)
            return (hj, cj)
        elif lextree[1] == -1:
            wordpieces = []
            for newpiece, values in lextree[0].items():
                wordpieces.append(self.get_lextree_encs_treelstm(values, newpiece))
            hs, cs = zip(*wordpieces)
            hs = torch.cat(hs, dim=0)
            cs = torch.cat(cs, dim=0)
            hj, cj = None, None
            if wordpiece is not None:
                ey = self.embed(to_device(self, torch.LongTensor([wordpiece])))
                ey = self.dropout_KB(ey)
                hj, cj = self.forward_treelstm_cell(hs, cs, ey)
                lextree.append(hj)
            return (hj, cj)

    def get_lextree_encs(self, lextree, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            ey = self.embed(to_device(self, torch.LongTensor([wordpiece])))
            ey = self.dropout_KB(ey)
            lextree.append(ey)
            return ey
        elif lextree[1] == -1 and lextree[0] != {}:
            wordpieces = []
            for newpiece, values in lextree[0].items():
                wordpieces.append(self.get_lextree_encs(values, newpiece))
            wordpiece_h = torch.cat(wordpieces, dim=0).sum(dim=0).unsqueeze(0)
            if wordpiece is not None:
                ey = self.embed(to_device(self, torch.LongTensor([wordpiece])))
                ey = self.dropout_KB(ey)
                wordpiece_h = self.recursive_proj(torch.cat([ey, wordpiece_h], dim=-1))
                wordpiece_h = torch.relu(wordpiece_h)
                # wordpiece_h = torch.sigmoid(wordpiece_h)
                lextree.append(wordpiece_h)
            return wordpiece_h

    def get_lextree_encs_gcn(self, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[0] == {} and wordpiece is not None:
            idx = len(embeddings)
            # ey = self.embed(to_device(self, torch.LongTensor([wordpiece])))
            ey = self.embed.weight[wordpiece].unsqueeze(0)
            embeddings.append(self.dropout_KB(ey))
            adjacency.append([idx])
            lextree.append([])
            lextree.append(idx)
            return idx
        elif lextree[0] != {}:
            ids = []
            idx = len(embeddings)
            if wordpiece is not None:
                # ey = self.embed(to_device(self, torch.LongTensor([wordpiece])))
                ey = self.embed.weight[wordpiece].unsqueeze(0)
                embeddings.append(self.dropout_KB(ey))
            for newpiece, values in lextree[0].items():
                ids.append(self.get_lextree_encs_gcn(values, embeddings, adjacency, newpiece))
            if wordpiece is not None:
                adjacency.append([idx] + ids)
            lextree.append(ids)
            lextree.append(idx)
            return idx

    def get_adjacency_mat(self, embeddings, adjacency):
        n_nodes = len(embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        adjacency_mat = embeddings.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0
        return embeddings, adjacency_mat

    def get_edge_ids(self, embeddings, adjacency):
        edge_ids = []
        embeddings = torch.cat(embeddings, dim=0)
        for node in adjacency:
            for neighbour in node:
                edge_ids.append([node[0], neighbour])
        return embeddings, to_device(self, torch.LongTensor(edge_ids)).t()

    def forward_gcn(self, lextree, embeddings, adjacency):
        # embeddings, adjacency_mat = self.get_adjacency_mat(embeddings, adjacency)
        n_nodes = len(embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        adjacency_mat = embeddings.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0
        # adjacency_mat = to_device(self, torch.Tensor(adjacency_mat))
        degrees = torch.diag(torch.sum(adjacency_mat, dim=-1) ** -0.5)
        nodes_encs = self.gcn_l1(embeddings)
        adjacency_mat = torch.einsum('ij,jk->ik', degrees, adjacency_mat)
        adjacency_mat = torch.einsum('ij,jk->ik', adjacency_mat, degrees)
        nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, nodes_encs))
        if self.treetype != 'gcn' and int(self.treetype[-1]) > 1:
            nodes_encs = self.gcn_l2(nodes_encs)
            nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, nodes_encs))
            if int(self.treetype[-1]) > 2:
                nodes_encs = self.gcn_l3(nodes_encs)
                nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, nodes_encs))
        return nodes_encs

    def forward_sage(self, lextree, embeddings, adjacency, layerid=1):
        # adjacency = adjacency - torch.eye(adjacency.size(0)).to(adjacency.device)
        nodes_encs = torch.relu(getattr(self, 'sage_pool_{}'.format(layerid))(embeddings))
        # nodes_encs = embeddings
        # nodes_encs = torch.cat([torch.ones_like(nodes_encs[:1])*-1e9, nodes_encs], dim=0)
        # index = torch.arange(1, adjacency.size(1) + 1).to(adjacency.device) * adjacency.long()
        # indexed = nodes_encs[index.flatten(), :].reshape(*adjacency.shape, nodes_encs.size(-1))
        # pooled_encs = indexed.max(dim=1)[0]
        # pooled_encs = pooled_encs * (adjacency.sum(dim=1, keepdim=True) != 0).float()

        pooled_encs = [0] * len(adjacency)
        for i, node in enumerate(adjacency):
            if len(node) > 1:
                candidates = nodes_encs[node[1:]]
                pooled_encs[node[0]] = candidates.max(dim=0)[0]
            elif len(node) == 1:
                pooled_encs[node[0]] = nodes_encs.new_zeros(nodes_encs.size(1))
        pooled_encs = torch.stack(pooled_encs)

        pooled_encs = torch.cat([embeddings, pooled_encs], dim=1)
        pooled_encs = torch.relu(getattr(self, 'sage_merge_{}'.format(layerid))(pooled_encs))
        return pooled_encs

    def fill_lextree_encs_gcn(self, lextree, nodes_encs, wordpiece=None):
        if lextree[0] == {} and wordpiece is not None:
            idx = lextree[4]
            # lextree.append(nodes_encs[idx])
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs_gcn(values, nodes_encs, newpiece)
            # lextree.append(nodes_encs[idx])
            lextree[3] = nodes_encs[idx].unsqueeze(0)

    def calc_ptr_loss(self, ptr_dist_all, model_dist, ptr_gen, ptr_gen_mask,
                      targets, ignore_idx, reduction_str, att_labs=None):
        ptr_dist = torch.cat(ptr_dist_all, dim=1)
        # Attention loss
        att_lab_loss = None
        if att_labs is not None:
            att_lab_loss = (-torch.log(ptr_dist+1e-9) * att_labs).sum() # / max(att_labs.sum(), 1.0)
        ptr_gen = torch.cat(ptr_gen, dim=1).masked_fill_(ptr_gen_mask.bool(), 0).view(-1, 1)
        # the gap to 1 is the prob for <unk>, which indicates not in the KB
        ptr_gen_complement = (ptr_dist[:,:,-1].view(targets.size(0), -1)) * ptr_gen
        model_dist = F.softmax(model_dist, dim=-1)
        p_final = ptr_dist[:,:,:-1].view(targets.size(0), -1) * ptr_gen + model_dist * (1 - ptr_gen + ptr_gen_complement)
        p_loss = F.nll_loss(torch.log(p_final+1e-9), targets,
                            ignore_index=ignore_idx, reduction=reduction_str)
        return p_loss, att_lab_loss, p_final, ptr_gen.view(ptr_gen_mask.size(0), -1)

    def encode_tree(self, prefixtree):
        if self.treetype == 'lstm':
            self.get_lextree_encs_treelstm(prefixtree)
        elif self.treetype.startswith('gcn'):
            embeddings, adjacency = [], []
            self.get_lextree_encs_gcn(prefixtree, embeddings, adjacency)
            nodes_encs = self.forward_gcn(prefixtree, embeddings, adjacency)
            self.fill_lextree_encs_gcn(prefixtree, nodes_encs)
        elif self.treetype.startswith('gat'):
            embeddings, adjacency = [], []
            self.get_lextree_encs_gcn(prefixtree, embeddings, adjacency)
            embeddings, adjacency_mat = self.get_adjacency_mat(embeddings, adjacency)
            # nodes_encs, edge_inds = self.get_edge_ids(embeddings, adjacency)
            nodes_encs, edge_inds = self.GAT((embeddings, adjacency_mat))
            # for l in range(len(self.nheads)):
            #     nodes_encs = getattr(self, 'gat_layer{}'.format(l))(nodes_encs, edge_inds)
            self.fill_lextree_encs_gcn(prefixtree, nodes_encs)
        elif self.treetype.startswith('sage'):
            embeddings, adjacency = [], []
            self.get_lextree_encs_gcn(prefixtree, embeddings, adjacency)
            embeddings = torch.cat(embeddings, dim=0)
            # embeddings, adjacency = self.get_adjacency_mat(embeddings, adjacency)
            nodes_encs = self.forward_sage(prefixtree, embeddings, adjacency)
            for i in range(int(self.treetype[-1])-1):
                nodes_encs = self.forward_sage(prefixtree, nodes_encs, adjacency, layerid=i+2)
            self.fill_lextree_encs_gcn(prefixtree, nodes_encs)
        else:
            self.get_lextree_encs(prefixtree)

    def get_last_word(self, charlist, split=False):
        starts = len(charlist) - 2
        char_tuple = [self.char_list[charlist[-1]]] if not split else [charlist[-1]]
        while starts > 0 and not self.char_list[charlist[starts]].endswith('▁'):
            char_tuple = [self.char_list[charlist[starts]] if not split else charlist[starts]] + char_tuple
            starts -= 1
        return tuple(char_tuple)

    def forward(self, hs_pad, hlens, ys_pad, strm_idx=0, lang_ids=None, meeting_info=None,
                att_labs=None):
        """Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
                                    [in multi-encoder case,
                                    list of torch.Tensor,
                                    [(B, Tmax_1, D), (B, Tmax_2, D), ..., ] ]
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
                                   [in multi-encoder case, list of torch.Tensor,
                                   [(B), (B), ..., ]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor
                                    (B, Lmax)
        :param int strm_idx: stream index indicates the index of decoding stream.
        :param torch.Tensor lang_ids: batch of target language id tensor (B, 1)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        """
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlens = [hlens]

        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # attention index for the attention module
        # in SPA (speaker parallel attention),
        # att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlens should be list of list of integer
        hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_ptrgen_pad = pad_list(ys_in, self.ignore_id)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        for idx in range(self.num_encs):
            logging.info(
                self.__class__.__name__
                + "Number of Encoder:{}; enc{}: input lengths: {}.".format(
                    self.num_encs, idx + 1, hlens[idx]
                )
            )
        logging.info(
            self.__class__.__name__
            + " output lengths: "
            + str([y.size(0) for y in ys_out])
        )

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        z_all = []
        if self.num_encs == 1:
            att_w = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # gs534 - get meeting KBs
        slotlist = False
        classposts = []
        if meeting_info is not None and self.epoch >= self.PtrSche:
            if meeting_info[2][0] is meeting_info[2][1]:
                self.encode_tree(meeting_info[2][0])
            else:
                slotlist = True
            #     for lextree in meeting_info[2]:
            #         if lextree[0] != {}:
            #             self.encode_tree(lextree)
            ptr_dist_all, p_gen_all = [], []
            trees = meeting_info[2]
            p_gen_mask_all = []

        # loop for an output sequence
        for i in six.moves.range(olength):
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](
                    hs_pad[0], hlens[0], self.dropout_dec[0](z_list[0]), att_w
                )
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](
                        hs_pad[idx],
                        hlens[idx],
                        self.dropout_dec[0](z_list[0]),
                        att_w_list[idx],
                    )
                hs_pad_han = torch.stack(att_c_list, dim=1)
                hlens_han = [self.num_encs] * len(ys_in)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                    hs_pad_han,
                    hlens_han,
                    self.dropout_dec[0](z_list[0]),
                    att_w_list[self.num_encs],
                )
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(" scheduled sampling ")
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(to_device(hs_pad[0], z_out)))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)

            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)

            # gs534 - Add KB
            if meeting_info is not None and self.meetingKB is not None and self.epoch >= self.PtrSche:
                factor = 0.0 if (self.epoch < self.PtrSche) else 1.0
                att_labs_i = att_labs[:,i] if att_labs is not None else None
                if self.ac_only:
                    query = self.dropout_KB(self.Qproj(ey))
                else:
                    # query = z_list[-1]
                    query = self.dropout_KB(self.Qproj(torch.cat([att_c, z_list[-1]], dim=-1)))
                if self.KBlextree and not slotlist:
                    if self.DBinput:
                        DBinput = self.DBembed((1-lex_masks[:, i, :-1].float()))
                    step_mask, trees, p_gen_mask, step_embs, back_transform = self.get_lextree_step_embs(
                        ys_ptrgen_pad[:, i], trees, meeting_info[2])
                    p_gen_mask_all.append(p_gen_mask)
                    KBembedding, ptr_dist = self.get_meetingKB_emb_map(query, step_mask, step_embs, back_transform)
                elif self.KBlextree:
                    KBembedding, ptr_dist, p_gen_mask, classpost = [], [], [], []
                    newtrees = []
                    for k, vy in enumerate(ys_ptrgen_pad[:, i]):
                        current_tree = trees[k]
                        vys = torch.stack([vy for count in range(len(current_tree))])
                        step_mask, tree_track, ptr_mask, step_embs, back_transform = self.get_lextree_step_embs(
                            vys, current_tree, meeting_info[2][k], clm=True)
                        newtrees.append(tree_track)
                        p_gen_mask.append(0 not in ptr_mask)
                        results = self.get_meetingKB_emb_map_forest(query[k], step_mask, step_embs, back_transform,
                            stacked=self.stacked)
                        KBembedding.append(results[0])
                        ptr_dist.append(results[1])
                        classpost.append(results[2])
                    trees = newtrees
                    KBembedding = torch.cat(KBembedding, dim=0)
                    classposts.append(torch.stack(classpost, dim=0).unsqueeze(1))
                    ptr_dist = torch.cat(ptr_dist, dim=0)
                    p_gen_mask_all.append(p_gen_mask)
                else:
                    print('To be implemented')
                # pointer generator distribution
                if self.PtrGen:
                    p_gen = torch.sigmoid(self.pointer_gate(torch.cat((z_list[-1] if self.ac_only else query,
                        KBembedding), dim=1)))
                    ptr_dist_all.append(ptr_dist.unsqueeze(1))
                    # Apply smoothing probability
                    if self.meetingKB.curriculum and self.meetingKB.fullepoch > 0:
                        smoothprob = min(1.0, self.epoch / self.meetingKB.fullepoch) * self.smoothprob
                    else:
                        smoothprob = self.smoothprob
                    # NOTE: factor controls whether to use PtrGen or not
                    p_gen_all.append(p_gen * smoothprob * factor)

            if self.useKBinput or self.DBinput:
                z_out = self.post_LSTM_proj(torch.cat(
                    (self.dropout_dec[-1](z_list[-1]), DBinput if self.DBinput else KBembedding), dim=-1))
            else:
                z_out = self.dropout_dec[-1](z_list[-1])

            if self.context_residual:
                z_all.append(
                    torch.cat((z_out, att_c), dim=-1)
                )  # utt x (zdim + hdim)
            else:
                z_all.append(z_out)  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1)
        if classposts != []:
            classposts = torch.cat(classposts, dim=1)
        # compute loss
        y_all = self.output(z_all.view(batch * olength, -1))
        if LooseVersion(torch.__version__) < LooseVersion("1.0"):
            reduction_str = "elementwise_mean"
        else:
            reduction_str = "mean"
        # gs534 - pointer generator loss
        pfinal = None
        KB_loss = None
        loss_sep = None
        ptr_gen = z_all.new_zeros(z_all.size(0), z_all.size(1))
        if meeting_info is not None and self.PtrGen and self.epoch >= self.PtrSche:
            ptr_mask = to_device(self, torch.tensor(p_gen_mask_all).byte()).t()
            self.loss, KB_loss, pfinal, ptr_gen = self.calc_ptr_loss(ptr_dist_all, y_all, p_gen_all, ptr_mask,
                ys_out_pad.view(-1), self.ignore_id, reduction_str, att_labs=att_labs)
        else:
            self.loss = F.cross_entropy(
                y_all,
                ys_out_pad.view(-1),
                ignore_index=self.ignore_id,
                reduction=reduction_str,
            )
        # compute perplexity
        ppl = math.exp(self.loss.item())
        # -1: eos, which is removed in the loss computation
        self.loss *= np.mean([len(x) for x in ys_in]) - 1
        # add KB loss
        # if KB_loss is not None and KB_loss != 0:
        #     nsamples = (att_labs.sum(dim=-1).sum(dim=-1) != 0).sum()
        #     KB_loss = KB_loss * self.KBlossfactor / nsamples
        #     self.loss += KB_loss

        acc = th_accuracy(pfinal if pfinal is not None else y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info("att loss:" + "".join(str(self.loss.item()).split("\n")))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(
                enumerate(ys_hat.detach().cpu().numpy()), ys_true.detach().cpu().numpy()
            ):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_device(hs_pad[0], torch.from_numpy(self.labeldist))
            loss_reg = -torch.sum(
                (F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0
            ) / len(ys_in)
            self.loss = (1.0 - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc, ppl, z_all, ptr_gen, classposts

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, strm_idx=0, meeting_info=None,
                       ranking_norm=False, fusion_lm_hidden=None, estlm=None, estlm_factor=0.0, estlm_hid=None,
                       dynamic_disc=False, sel_lm=None, topk=1, prev_hid=None, classlm=False, slotlist=False):
        """beam search implementation

        :param torch.Tensor h: encoder hidden state (T, eprojs)
                                [in multi-encoder case, list of torch.Tensor,
                                [(T1, eprojs), (T2, eprojs), ...] ]
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
                                [in multi-encoder case, list of torch.Tensor,
                                [(T1, odim), (T2, odim), ...] ]
        :param Namespace recog_args: argument Namespace containing options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :param int strm_idx:
            stream index for speaker parallel attention in multi-speaker case
        :return: N-best decoding results
        :rtype: list of dicts
        """
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            h = [h]
            lpz = [lpz]
        if self.num_encs > 1 and lpz is None:
            lpz = [lpz] * self.num_encs

        for idx in range(self.num_encs):
            logging.info(
                "Number of Encoder:{}; enc{}: input lengths: {}.".format(
                    self.num_encs, idx + 1, h[0].size(0)
                )
            )
        att_idx = min(strm_idx, len(self.att) - 1)
        # initialization
        c_list = [self.zero_state(h[0].unsqueeze(0))]
        z_list = [self.zero_state(h[0].unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h[0].unsqueeze(0)))
            z_list.append(self.zero_state(h[0].unsqueeze(0)))
        # forward ILME
        c_list_ilme = [self.zero_state(h[0].unsqueeze(0))]
        z_list_ilme = [self.zero_state(h[0].unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list_ilme.append(self.zero_state(h[0].unsqueeze(0)))
            z_list_ilme.append(self.zero_state(h[0].unsqueeze(0)))
        if self.num_encs == 1:
            a = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = getattr(recog_args, "ctc_weight", False)  # for NMT

        if lpz[0] is not None and self.num_encs > 1:
            # weights-ctc,
            # e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
            weights_ctc_dec = recog_args.weights_ctc_dec / np.sum(
                recog_args.weights_ctc_dec
            )  # normalize
            logging.info(
                "ctc weights (decoding): " + " ".join([str(x) for x in weights_ctc_dec])
            )
        else:
            weights_ctc_dec = [1.0]

        # preprate sos
        if self.replace_sos and recog_args.tgt_lang:
            y = char_list.index(recog_args.tgt_lang)
        else:
            y = self.sos
        logging.info("<sos> index: " + str(y))
        logging.info("<sos> mark: " + char_list[y])
        vy = h[0].new_zeros(1).long()

        maxlen = np.amin([h[idx].size(0) for idx in range(self.num_encs)])
        if recog_args.maxlenratio != 0:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * maxlen))
        minlen = int(recog_args.minlenratio * maxlen)
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {
                "score": 0.0,
                "yseq": [y],
                "c_prev": c_list,
                "z_prev": z_list,
                "a_prev": a,
                "rnnlm_prev": fusion_lm_hidden,
                "a_accum": to_device(self, torch.zeros(h[0].size(0), dtype=torch.float32)),
                "final_score": 0.0,
                "hidden_states": [],
                "p_gen": []
            }
        else:
            hyp = {
                "score": 0.0,
                "yseq": [y],
                "c_prev": c_list,
                "z_prev": z_list,
                "a_prev": a,
                "a_accum": to_device(self, torch.zeros(h[0].size(0), dtype=torch.float32)),
                "final_score": 0.0,
                "hidden_states": [],
                "p_gen": []
            }
        # forward ILME
        hyp['c_prev_ilme'] = c_list_ilme
        hyp['z_prev_ilme'] = z_list_ilme
        # gs534 - get meeting KBs
        if meeting_info is not None and meeting_info != []:
            if self.KBlextree:
                if isinstance(meeting_info[2][0], str):
                    hyp['lextree'] = self.meetingKB.meetinglextree[meeting_info[2][0]].copy()
                else:
                    hyp['lextree'] = meeting_info[2][0]
                if sel_lm is None: # and not slotlist:
                    self.encode_tree(hyp['lextree'])
            KBembedding = h[0].new_zeros(1, self.dunits)
            if sel_lm is not None:
                hyp['sel_lm_hidden'] = sel_lm.init_hidden(1) if prev_hid is None else prev_hid
                if classlm:
                    hyp['lextree'] = [hyp['lextree'] for i in range(topk)]

        if estlm is not None:
            hyp['estlm_prev'] = estlm.init_hidden(1) if estlm_hid is None else estlm_hid

        if lpz[0] is not None:
            ctc_prefix_score = [
                CTCPrefixScore(lpz[idx].detach().numpy(), 0, self.eos, np)
                for idx in range(self.num_encs)
            ]
            hyp["ctc_state_prev"] = [
                ctc_prefix_score[idx].initial_state() for idx in range(self.num_encs)
            ]
            hyp["ctc_score_prev"] = [0.0] * self.num_encs
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz[0].shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz[0].shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]
                ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
                if self.num_encs == 1:
                    att_c, att_w = self.att[att_idx](
                        h[0].unsqueeze(0),
                        [h[0].size(0)],
                        self.dropout_dec[0](hyp["z_prev"][0]),
                        hyp["a_prev"],
                    )
                    hyp["a_accum"] += to_device(self, att_w[0])
                else:
                    for idx in range(self.num_encs):
                        att_c_list[idx], att_w_list[idx] = self.att[idx](
                            h[idx].unsqueeze(0),
                            [h[idx].size(0)],
                            self.dropout_dec[0](hyp["z_prev"][0]),
                            hyp["a_prev"][idx],
                        )
                    h_han = torch.stack(att_c_list, dim=1)
                    att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                        h_han,
                        [self.num_encs],
                        self.dropout_dec[0](hyp["z_prev"][0]),
                        hyp["a_prev"][self.num_encs],
                    )
                    hyp["a_accum"] += att_w_list[0][0]
                ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
                # forward ILME
                ey_ilme = torch.cat((self.embed(vy), att_c*0), dim=1)  # utt(1) x (zdim + hdim)
                z_list, c_list = self.rnn_forward(
                    ey, z_list, c_list, hyp["z_prev"], hyp["c_prev"]
                )
                # forward ILME
                z_list_ilme, c_list_ilme = self.rnn_forward(
                    ey_ilme, z_list_ilme, c_list_ilme, hyp["z_prev_ilme"], hyp["c_prev_ilme"]
                )

                # gs534 - meeting level KB
                if self.meetingKB is not None and meeting_info != []:
                    # select KB entries and organise reset tree
                    if sel_lm is not None and (vy[0].data == self.eos or self.char_list[vy].endswith('▁')):
                        if vy[0].data == self.eos:
                            thisword = self.meetingKB.vocab.get_idx('<eos>')
                        else:
                            thisword = self.meetingKB.vocab.get_idx(self.get_last_word(hyp['yseq']))
                        lmout, hyp['sel_lm_hidden'], class_out = sel_lm(
                            torch.LongTensor([[thisword]]), hyp['sel_lm_hidden'])
                        # lmout = torch.log_softmax(lmout.squeeze(0).squeeze(0), dim=-1) - unigram_p * self.meetingKB.unigram_dist
                        new_select = []
                        if classlm:
                            class_out = class_out.squeeze(0).squeeze(0)
                            class_probs, classes = torch.topk(class_out, topk, dim=0)
                            # option: combine all classes at once
                            # for classid in classes:
                            #     for word in self.meetingKB.classdict[self.meetingKB.classorder[classid]]:
                            #         wordid = self.meetingKB.rarewords_word[word]
                            #         if wordid not in new_select:
                            #             new_select.append(wordid)
                            # reset_lextree = self.meetingKB.get_tree_from_inds(new_select, extra=meeting_info[1])
                            # self.encode_tree(reset_lextree)
                            reset_lextree = self.meetingKB.get_classed_trees(classes)
                        else:
                            lmout = lmout.squeeze(0).squeeze(0)
                            new_lm_output = lmout[meeting_info[0][0]]
                            values, new_select = torch.topk(new_lm_output, topk, dim=0)
                            reset_lextree = self.meetingKB.get_tree_from_inds(new_select, extra=meeting_info[1])
                            self.encode_tree(reset_lextree)
                    else:
                        meeting = meeting_info[2][0]
                        reset_lextree = self.meetingKB.meetinglextree[meeting] if isinstance(meeting, str) else meeting

                    if self.KBlextree:
                        if self.ac_only:
                            query = self.dropout_KB(self.Qproj(ey))
                            # query = self.dropout_KB(self.Qproj(torch.cat([self.dropout_emb(self.embed(vy))*0, att_c], dim=1)))
                        else:
                            # query = z_list[-1]
                            query = self.dropout_KB(self.Qproj(torch.cat([att_c, z_list[-1]], dim=-1)))
                        # Use deep biasing
                        if self.DBinput:
                            # NOTE: change it back if does not work
                            DBinput = self.DBembed((1-lex_mask[:-1].unsqueeze(0).float()))

                        if (classlm and new_select == []): # or slotlist:
                            vys = torch.stack([vy for count in range(len(hyp['lextree']))])
                            step_mask, tree_track, ptr_mask, step_embs, back_transform = self.get_lextree_step_embs(
                                vys, hyp['lextree'], reset_lextree, clm=True)
                            inKB = 0 in ptr_mask
                            KBembedding, ptr_dist, _ = self.get_meetingKB_emb_map_forest(query, step_mask, step_embs, back_transform)
                        else:
                            step_mask, tree_track, inKB, step_embs, back_transform = self.get_lextree_step_embs_inference(
                                vy, hyp['lextree'], reset_lextree)
                            KBembedding, ptr_dist = self.get_meetingKB_emb_map(query, step_mask, step_embs, back_transform)

                        # debug - gs534
                        if not inKB:
                            p_gen = 0
                        else:
                            history_repre = z_list[-1] if self.ac_only else query
                            p_gen = torch.sigmoid(self.pointer_gate(torch.cat([history_repre, KBembedding], dim=1)))
                    else:
                        KBembedding, ptr_dist = self.get_meetingKB_emb(hyp['z_prev'][0], meeting_KB, meeting_info[1])

                # get nbest local scores and their ids
                if self.context_residual:
                    if self.useKBinput or self.DBinput:
                        decoder_output = torch.cat([self.dropout_dec[-1](z_list[-1]),
                            DBinput if self.DBinput else KBembedding], dim=-1)
                        decoder_output = self.post_LSTM_proj(decoder_output)
                        decoder_output = torch.cat((decoder_output, att_c), dim=-1)
                    else:
                        decoder_output = torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                    # forward ILME
                    decoder_output_ilme = torch.cat((self.dropout_dec[-1](z_list_ilme[-1]), att_c*0), dim=-1)
                elif self.useKBinput or self.DBinput:
                    decoder_output = torch.cat([self.dropout_dec[-1](z_list[-1]), DBinput], dim=-1)
                    decoder_output = self.post_LSTM_proj(decoder_output)
                    # forward ILME
                    decoder_output_ilme = self.dropout_dec[-1](z_list_ilme[-1])
                else:
                    decoder_output = self.dropout_dec[-1](z_list[-1])
                    # forward ILME
                    decoder_output_ilme = self.dropout_dec[-1](z_list_ilme[-1])
                logits = self.output(decoder_output)
                # forward ILME
                logits_ilme = self.output(decoder_output_ilme)

                # gs534 - pointer generator
                if not self.PtrGen or meeting_info == []:
                    local_att_scores = F.log_softmax(logits, dim=1)
                else:
                    p_gen = p_gen * self.smoothprob
                    if dynamic_disc and estlm and rnnlm:
                        model_dist = F.log_softmax(logits, dim=-1)
                        # Internal LM
                        estlm_scores, hyp['estlm_prev'] = estlm(vy.view(1, -1), hyp['estlm_prev'])
                        estlm_scores = estlm_scores.squeeze(0)
                        # Target LM
                        rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                        
                        # estlm_scores_masked = F.softmax(estlm_scores.masked_fill(
                        #     lex_mask[:-1].view(1, -1).bool(), -1e9), dim=-1)
                        # local_lm_scores_masked = F.softmax(estlm_scores.masked_fill(
                        #     lex_mask[:-1].view(1, -1).bool(), -1e9), dim=-1)
                        estlm_scores_masked = F.softmax(estlm_scores, dim=-1)
                        local_lm_scores_masked = F.softmax(local_lm_scores, dim=-1)

                        estlm_scores = F.log_softmax(estlm_scores, dim=-1)
                        local_lm_scores = F.log_softmax(local_lm_scores, dim=-1)
                        model_disc_dist = model_dist - estlm_factor * estlm_scores + recog_args.lm_weight * local_lm_scores
                        model_dist = torch.exp(model_disc_dist)
                        
                        # For Ptr dist
                        # estlm_scores_masked.masked_fill_(lex_mask[:-1].view(1, -1).bool(), 1)
                        estlm_multiply = (local_lm_scores_masked ** 0.4) / (estlm_scores_masked ** (0.3)) 
                        ptr_dist[:,:-1] = ptr_dist[:,:-1] * estlm_multiply
                        # ptr_dist[:,:-1] = ptr_dist[:,:-1] * (local_lm_scores_masked ** (recog_args.lm_weight - 0.2))
                    else:
                        model_dist = F.softmax(logits, dim=-1)
                    # if hyp['yseq'] == [200, 144, 65, 1, 103, 55, 167, 43, 136, 136, 68, 152, 180, 1, 132, 106, 7, 69, 136]:
                    #     import pdb; pdb.set_trace()
                    ptr_gen_complement = ptr_dist[:,-1] * p_gen
                    # print(self.char_list[hyp['yseq'][-1]])
                    # print(ptr_dist[:,:-1].sum()*p_gen)
                    local_att_scores = torch.log(ptr_dist[:,:-1] * p_gen + model_dist * (1 - p_gen + ptr_gen_complement))

                # subtract ILME scores
                if getattr(recog_args, 'ilme', 0) > 0:
                    ilme_scores = F.log_softmax(logits_ilme, dim=1)
                    local_att_scores = local_att_scores - getattr(recog_args, 'ilme', 0) * ilme_scores

                if estlm and not dynamic_disc:
                    estlm_scores, hyp['estlm_prev'] = estlm(vy.view(1, -1), hyp['estlm_prev'])
                    estlm_scores = F.log_softmax(estlm_scores, dim=-1)
                    local_att_scores -= estlm_factor * estlm_scores.squeeze(0)

                if rnnlm and not dynamic_disc:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz[0] is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = (
                        [None] * self.num_encs,
                        [None] * self.num_encs,
                    )
                    for idx in range(self.num_encs):
                        ctc_scores[idx], ctc_states[idx] = ctc_prefix_score[idx](
                            hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"][idx]
                        )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ]
                    if self.num_encs == 1:
                        local_scores += ctc_weight * torch.from_numpy(
                            ctc_scores[0] - hyp["ctc_score_prev"][0]
                        )
                    else:
                        for idx in range(self.num_encs):
                            local_scores += (
                                ctc_weight
                                * weights_ctc_dec[idx]
                                * torch.from_numpy(
                                    ctc_scores[idx] - hyp["ctc_score_prev"][idx]
                                )
                            )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp["z_prev"] = z_list[:]
                    new_hyp["c_prev"] = c_list[:]
                    # forward ILME
                    new_hyp["z_prev_ilme"] = z_list_ilme[:]
                    new_hyp["c_prev_ilme"] = c_list_ilme[:]
                    if self.num_encs == 1:
                        new_hyp["a_prev"] = att_w[:]
                    else:
                        new_hyp["a_prev"] = [
                            att_w_list[idx][:] for idx in range(self.num_encs + 1)
                        ]
                    new_hyp["a_accum"] = hyp["a_accum"].clone().detach()
                    new_hyp["score"] = hyp["score"] + local_best_scores[0, j]
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz[0] is not None:
                        new_hyp["ctc_state_prev"] = [
                            ctc_states[idx][joint_best_ids[0, j]]
                            for idx in range(self.num_encs)
                        ]
                        new_hyp["ctc_score_prev"] = [
                            ctc_scores[idx][joint_best_ids[0, j]]
                            for idx in range(self.num_encs)
                        ]
                    # gs534 - lextree
                    if self.KBlextree and meeting_info != []:
                        new_hyp['lextree'] = tree_track.copy()
                        new_hyp['p_gen'] = hyp['p_gen'] + [(ptr_dist.sum()*p_gen).item()]
                    else:
                        new_hyp['p_gen'] = hyp['p_gen'] + [0]
                    if sel_lm is not None:
                        new_hyp['sel_lm_hidden'] = (hyp['sel_lm_hidden'][0].clone(),
                                                    hyp['sel_lm_hidden'][1].clone())

                    if estlm:
                        new_hyp['estlm_prev'] = (hyp['estlm_prev'][0].clone(),
                                                 hyp['estlm_prev'][1].clone())
                    new_hyp['hidden_states'] = hyp['hidden_states'] + [decoder_output]

                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypotheses: " + str(len(hyps)))
            logging.debug(
                "best hypo: "
                + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
            )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    if hyp["yseq"][-1] != self.eos:
                        hyp["yseq"].append(self.eos)

            # add ended hypotheses to a final list,
            # and removed them from current hypotheses
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        # add coverage penalty to compute hyp final score
                        coverage_score = torch.clamp(hyp["a_accum"], 0.001, 0.5).log().sum()
                        hyp["final_score"] = (
                            hyp["score"] + recog_args.coverage_penalty * coverage_score)
                        # forbids hyp that emits eos below threshold
                        if hyp["score"] > hyps[0]["score"] - recog_args.eos_max_logit_delta:
                            ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remaining hypotheses: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            for hyp in hyps:
                logging.debug(
                    "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                )

            logging.debug("number of ended hypotheses: " + str(len(ended_hyps)))

        if ranking_norm:
            ranking_fn = lambda x: x["final_score"] / (len(x["yseq"]) - 1)
        else:
            ranking_fn = lambda x: x["final_score"]
        nbest_hyps = sorted(ended_hyps, key=ranking_fn, reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        best_fusion_hid = None
        best_est_hid = None
        if 'rnnlm_prev' in nbest_hyps[0]:
            best_fusion_hid = nbest_hyps[0]['rnnlm_prev']
        if 'estlm_prev' in nbest_hyps[0]:
            best_est_hid = nbest_hyps[0]['estlm_prev']
        best_hid = None
        if 'sel_lm_hidden' in nbest_hyps[0]:
            best_hid = nbest_hyps[0]['sel_lm_hidden']

        # check number of hypotheses
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, "
                "perform recognition again with smaller minlenratio."
            )
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            if self.num_encs == 1:
                return self.recognize_beam(h[0], lpz[0], recog_args, char_list, rnnlm)
            else:
                return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info("final log probability: " + str(nbest_hyps[0]["final_score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / (len(nbest_hyps[0]["yseq"]) - 1))
        )

        # remove sos
        return nbest_hyps, best_fusion_hid, best_est_hid, best_hid

    def recognize_beam_batch(
        self,
        h,
        hlens,
        lpz,
        recog_args,
        char_list,
        rnnlm=None,
        normalize_score=True,
        strm_idx=0,
        lang_ids=None,
    ):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            h = [h]
            hlens = [hlens]
            lpz = [lpz]
        if self.num_encs > 1 and lpz is None:
            lpz = [lpz] * self.num_encs

        att_idx = min(strm_idx, len(self.att) - 1)
        for idx in range(self.num_encs):
            logging.info(
                "Number of Encoder:{}; enc{}: input lengths: {}.".format(
                    self.num_encs, idx + 1, h[idx].size(1)
                )
            )
            h[idx] = mask_by_length(h[idx], hlens[idx], 0.0)

        # search params
        batch = len(hlens[0])
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = getattr(recog_args, "ctc_weight", 0)  # for NMT
        att_weight = 1.0 - ctc_weight
        ctc_margin = getattr(
            recog_args, "ctc_window_margin", 0
        )  # use getattr to keep compatibility
        # weights-ctc,
        # e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
        if lpz[0] is not None and self.num_encs > 1:
            weights_ctc_dec = recog_args.weights_ctc_dec / np.sum(
                recog_args.weights_ctc_dec
            )  # normalize
            logging.info(
                "ctc weights (decoding): " + " ".join([str(x) for x in weights_ctc_dec])
            )
        else:
            weights_ctc_dec = [1.0]

        n_bb = batch * beam
        pad_b = to_device(h[0], torch.arange(batch) * beam).view(-1, 1)

        max_hlen = np.amin([max(hlens[idx]) for idx in range(self.num_encs)])
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialization
        c_prev = [
            to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)
        ]
        z_prev = [
            to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)
        ]
        c_list = [
            to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)
        ]
        z_list = [
            to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)
        ]
        vscores = to_device(h[0], torch.zeros(batch, beam))

        rnnlm_state = None
        if self.num_encs == 1:
            a_prev = [None]
            att_w_list, ctc_scorer, ctc_state = [None], [None], [None]
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a_prev = [None] * (self.num_encs + 1)  # atts + han
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            ctc_scorer, ctc_state = [None] * (self.num_encs), [None] * (self.num_encs)
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        if self.replace_sos and recog_args.tgt_lang:
            logging.info("<sos> index: " + str(char_list.index(recog_args.tgt_lang)))
            logging.info("<sos> mark: " + recog_args.tgt_lang)
            yseq = [
                [char_list.index(recog_args.tgt_lang)] for _ in six.moves.range(n_bb)
            ]
        elif lang_ids is not None:
            # NOTE: used for evaluation during training
            yseq = [
                [lang_ids[b // recog_args.beam_size]] for b in six.moves.range(n_bb)
            ]
        else:
            logging.info("<sos> index: " + str(self.sos))
            logging.info("<sos> mark: " + char_list[self.sos])
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = [
            hlens[idx].repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
            for idx in range(self.num_encs)
        ]
        exp_hlens = [exp_hlens[idx].view(-1).tolist() for idx in range(self.num_encs)]
        exp_h = [
            h[idx].unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
            for idx in range(self.num_encs)
        ]
        exp_h = [
            exp_h[idx].view(n_bb, h[idx].size()[1], h[idx].size()[2])
            for idx in range(self.num_encs)
        ]

        if lpz[0] is not None:
            scoring_num = min(
                int(beam * CTC_SCORING_RATIO)
                if att_weight > 0.0 and not lpz[0].is_cuda
                else 0,
                lpz[0].size(-1),
            )
            ctc_scorer = [
                CTCPrefixScoreTH(
                    lpz[idx],
                    hlens[idx],
                    0,
                    self.eos,
                    margin=ctc_margin,
                )
                for idx in range(self.num_encs)
            ]

        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            vy = to_device(h[0], torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout_emb(self.embed(vy))
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](
                    exp_h[0], exp_hlens[0], self.dropout_dec[0](z_prev[0]), a_prev[0]
                )
                att_w_list = [att_w]
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](
                        exp_h[idx],
                        exp_hlens[idx],
                        self.dropout_dec[0](z_prev[0]),
                        a_prev[idx],
                    )
                exp_h_han = torch.stack(att_c_list, dim=1)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                    exp_h_han,
                    [self.num_encs] * n_bb,
                    self.dropout_dec[0](z_prev[0]),
                    a_prev[self.num_encs],
                )
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            if self.context_residual:
                logits = self.output(
                    torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                )
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores

            # ctc
            if ctc_scorer[0]:
                local_scores[:, 0] = self.logzero  # avoid choosing blank
                part_ids = (
                    torch.topk(local_scores, scoring_num, dim=-1)[1]
                    if scoring_num > 0
                    else None
                )
                for idx in range(self.num_encs):
                    att_w = att_w_list[idx]
                    att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
                    local_ctc_scores, ctc_state[idx] = ctc_scorer[idx](
                        yseq, ctc_state[idx], part_ids, att_w_
                    )
                    local_scores = (
                        local_scores
                        + ctc_weight * weights_ctc_dec[idx] * local_ctc_scores
                    )

            local_scores = local_scores.view(batch, beam, self.odim)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = (
                torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            )
            accum_padded_beam_ids = (
                (accum_best_ids // self.odim + pad_b).view(-1).data.cpu().tolist()
            )

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(h[0], torch.LongTensor(accum_padded_beam_ids))

            a_prev = []
            num_atts = self.num_encs if self.num_encs == 1 else self.num_encs + 1
            for idx in range(num_atts):
                if isinstance(att_w_list[idx], torch.Tensor):
                    _a_prev = torch.index_select(
                        att_w_list[idx].view(n_bb, *att_w_list[idx].shape[1:]), 0, vidx
                    )
                elif isinstance(att_w_list[idx], list):
                    # handle the case of multi-head attention
                    _a_prev = [
                        torch.index_select(att_w_one.view(n_bb, -1), 0, vidx)
                        for att_w_one in att_w_list[idx]
                    ]
                else:
                    # handle the case of location_recurrent when return is a tuple
                    _a_prev_ = torch.index_select(
                        att_w_list[idx][0].view(n_bb, -1), 0, vidx
                    )
                    _h_prev_ = torch.index_select(
                        att_w_list[idx][1][0].view(n_bb, -1), 0, vidx
                    )
                    _c_prev_ = torch.index_select(
                        att_w_list[idx][1][1].view(n_bb, -1), 0, vidx
                    )
                    _a_prev = (_a_prev_, (_h_prev_, _c_prev_))
                a_prev.append(_a_prev)
            z_prev = [
                torch.index_select(z_list[li].view(n_bb, -1), 0, vidx)
                for li in range(self.dlayers)
            ]
            c_prev = [
                torch.index_select(c_list[li].view(n_bb, -1), 0, vidx)
                for li in range(self.dlayers)
            ]

            # pick ended hyps
            if i >= minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        _vscore = None
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            if len(yk) <= min(
                                hlens[idx][samp_i] for idx in range(self.num_encs)
                            ):
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                        elif i == maxlen - 1:
                            yk = yseq[k][:]
                            _vscore = vscores[samp_i][beam_j] + penalty_i
                        if _vscore:
                            yk.append(self.eos)
                            if rnnlm:
                                _vscore += recog_args.lm_weight * rnnlm.final(
                                    rnnlm_state, index=k
                                )
                            _score = _vscore.data.cpu().numpy()
                            ended_hyps[samp_i].append(
                                {"yseq": yk, "vscore": _vscore, "score": _score}
                            )
                        k = k + 1

            # end detection
            stop_search = [
                stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                for samp_i in six.moves.range(batch)
            ]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer[0]:
                for idx in range(self.num_encs):
                    ctc_state[idx] = ctc_scorer[idx].index_select_state(
                        ctc_state[idx], accum_best_ids
                    )

        torch.cuda.empty_cache()

        dummy_hyps = [
            {"yseq": [self.sos, self.eos], "score": np.array([-float("inf")])}
        ]
        ended_hyps = [
            ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
            for samp_i in six.moves.range(batch)
        ]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x["score"] /= len(x["yseq"])

        nbest_hyps = [
            sorted(ended_hyps[samp_i], key=lambda x: x["score"], reverse=True)[
                : min(len(ended_hyps[samp_i]), recog_args.nbest)
            ]
            for samp_i in six.moves.range(batch)
        ]

        return nbest_hyps

    # Replication of batch decoding for MBR training
    def batch_decode_nbest(self, h, hlens, beam=1, nbest=1, maxlen=0, normalize_score=True,
                           strm_idx=0, meeting_info=None):
        # ONLY CONSIDER SINGLE ENCODER CASE HERE
        att_idx = min(strm_idx, len(self.att) - 1)
        # h = mask_by_length(h, hlens, 0.0)

        # search params
        batch = len(hlens)

        n_bb = batch * beam
        pad_b = to_device(self, torch.arange(batch) * beam).view(-1, 1)

        max_hlen = max(hlens)
        maxlen = maxlen if maxlen > 0 else min(max_hlen, 400) # TODO: should be set to a more sensible number
        minlen = 3 # int(recog_args.minlenratio * max_hlen)

        # initialization
        c_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_device(self, torch.zeros(batch, beam))

        a_prev = None
        self.att[att_idx].reset()  # reset pre-computation of h
        yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2])

        # gs534 - meeting-level KB
        if meeting_info is not None:
            tree_track = []
            lex_mask_mat = to_device(self, torch.ones(len(self.char_list) + 1))
            if self.KBlextree:
                tree_track = [meeting_info[2][0].copy()] * n_bb
                lex_mask_mat[list(meeting_info[2][0][0].keys())] = 0
                lex_mask_mat[-1] = 0
                inKB_mat = to_device(self, torch.tensor([0] * n_bb))
                lex_mask_mat = lex_mask_mat.unsqueeze(0).repeat(n_bb, 1).byte()

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            vy = to_device(self, torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout_emb(self.embed(vy))
            att_c, att_w = self.att[att_idx](exp_h, exp_hlens, self.dropout_dec[0](z_prev[0]), a_prev)
            ey = torch.cat((ey, att_c), dim=1)
            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)

            if self.meetingKB is not None and meeting_info != []:
                if self.KBlextree:
                    if self.ac_only:
                        query = self.dropout_KB(self.Qproj(ey))
                    else:
                        # query = z_list[-1]
                        query = self.dropout_KB(self.Qproj(torch.cat([att_c, z_list[-1]], dim=-1)))
                    KBembedding, ptr_dist = self.get_meetingKB_emb(query, lex_mask_mat)
                    history_repre = z_list[-1] if self.ac_only else query
                    p_gen = torch.sigmoid(self.pointer_gate(torch.cat((history_repre, KBembedding), dim=1)))
                else:
                    KBembedding, ptr_dist = self.get_meetingKB_emb(z_list[-1], meeting_KB, lex_mask_mat)

            if self.context_residual:
                logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))

            # gs534 - TCPGen
            if self.meetingKB is not None and self.PtrGen:
                p_gen = p_gen.masked_fill(inKB_mat.view(-1, 1).bool(), 0) * self.smoothprob
                model_dist = F.softmax(logits, dim=-1)
                # (n_bb, KBsize) * (n_bb, KBsize, odim) -> (n_bb, odim)
                ptr_gen_complement = ptr_dist[:,-1:] * p_gen
                local_scores = torch.log(ptr_dist[:,:-1] * p_gen + model_dist * (1 - p_gen + ptr_gen_complement))
            else:
                local_scores = F.log_softmax(logits, dim=1)

            local_scores = local_scores.view(batch, beam, self.odim)
            if i == 0:
                mask_other_beams = torch.cat(
                    [local_scores.new_zeros(batch, 1, self.odim),
                     local_scores.new_ones(batch, beam-1, self.odim)], dim=1).byte()
                local_scores = local_scores.masked_fill(mask_other_beams.bool(), self.logzero)

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim, rounding_mode='floor') + pad_b).view(-1).data.cpu().tolist()

            # gs534 - get meeting KBs for each step
            if meeting_info is not None and self.KBlextree:
                new_tree_track = []
                lex_mask_mat = []
                inKB_mat = []
                for batchbeam in range(n_bb):
                    b_ids = accum_padded_beam_ids[batchbeam]
                    new_tree, lex_mask, inKB = self.meeting_lextree_step(
                        accum_odim_ids[batchbeam], tree_track[b_ids].copy(), meeting_info[2][0])
                    new_tree_track.append(new_tree)
                    lex_mask_mat.append(lex_mask)
                    inKB_mat.append(0 if inKB else 1)
                lex_mask_mat = to_device(self, torch.stack(lex_mask_mat)).byte()
                inKB_mat = to_device(self, torch.tensor(inKB_mat))
                tree_track = new_tree_track

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids))

            if isinstance(att_w, torch.Tensor):
                _a_prev = torch.index_select(att_w.view(n_bb, *att_w.shape[1:]), 0, vidx)
            elif isinstance(att_w, list):
                # handle the case of multi-head attention
                _a_prev = [torch.index_select(att_w_one.view(n_bb, -1), 0, vidx) for att_w_one in att_w]
            else:
                # handle the case of location_recurrent when return is a tuple
                _a_prev_ = torch.index_select(att_w[0].view(n_bb, -1), 0, vidx)
                _h_prev_ = torch.index_select(att_w[1][0].view(n_bb, -1), 0, vidx)
                _c_prev_ = torch.index_select(att_w[1][1].view(n_bb, -1), 0, vidx)
                _a_prev = (_a_prev_, (_h_prev_, _c_prev_))
            a_prev = _a_prev
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

            # pick ended hyps
            if i > minlen:
                k = 0
                # penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            yk.append(self.eos)
                            if len(yk) < hlens[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] # + penalty_i
                                _score = _vscore.data.cpu().numpy()
                                ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})
                        k = k + 1
            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

        torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'vscore': h.new_zeros(1)[0]-1e-9, 'score': np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x['score'] /= len(x['yseq'])

        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), nbest)]
                      for samp_i in six.moves.range(batch)]

        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad, strm_idx=0, lang_ids=None):
        """Calculate all of attentions

        :param torch.Tensor hs_pad: batch of padded hidden state sequences
                                    (B, Tmax, D)
                                    in multi-encoder case, list of torch.Tensor,
                                    [(B, Tmax_1, D), (B, Tmax_2, D), ..., ] ]
        :param torch.Tensor hlen: batch of lengths of hidden state sequences (B)
                                    [in multi-encoder case, list of torch.Tensor,
                                    [(B), (B), ..., ]
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx:
            stream index for parallel speaker attention in multi-speaker case
        :param torch.Tensor lang_ids: batch of target language id tensor (B, 1)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) multi-encoder case =>
                [(B, Lmax, Tmax1), (B, Lmax, Tmax2), ..., (B, Lmax, NumEncs)]
            3) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlen = [hlen]

        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlen should be list of list of integer
        hlen = [list(map(int, hlen[idx])) for idx in range(self.num_encs)]

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        att_ws = []
        if self.num_encs == 1:
            att_w = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](
                    hs_pad[0], hlen[0], self.dropout_dec[0](z_list[0]), att_w
                )
                att_ws.append(att_w)
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](
                        hs_pad[idx],
                        hlen[idx],
                        self.dropout_dec[0](z_list[0]),
                        att_w_list[idx],
                    )
                hs_pad_han = torch.stack(att_c_list, dim=1)
                hlen_han = [self.num_encs] * len(ys_in)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                    hs_pad_han,
                    hlen_han,
                    self.dropout_dec[0](z_list[0]),
                    att_w_list[self.num_encs],
                )
                att_ws.append(att_w_list.copy())
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)

        if self.num_encs == 1:
            # convert to numpy array with the shape (B, Lmax, Tmax)
            att_ws = att_to_numpy(att_ws, self.att[att_idx])
        else:
            _att_ws = []
            for idx, ws in enumerate(zip(*att_ws)):
                ws = att_to_numpy(ws, self.att[idx])
                _att_ws.append(ws)
            att_ws = _att_ws
        return att_ws

    @staticmethod
    def _get_last_yseq(exp_yseq):
        last = []
        for y_seq in exp_yseq:
            last.append(y_seq[-1])
        return last

    @staticmethod
    def _append_ids(yseq, ids):
        if isinstance(ids, list):
            for i, j in enumerate(ids):
                yseq[i].append(j)
        else:
            for i in range(len(yseq)):
                yseq[i].append(ids)
        return yseq

    @staticmethod
    def _index_select_list(yseq, lst):
        new_yseq = []
        for i in lst:
            new_yseq.append(yseq[i][:])
        return new_yseq

    @staticmethod
    def _index_select_lm_state(rnnlm_state, dim, vidx):
        if isinstance(rnnlm_state, dict):
            new_state = {}
            for k, v in rnnlm_state.items():
                new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
        elif isinstance(rnnlm_state, list):
            new_state = []
            for i in vidx:
                new_state.append(rnnlm_state[int(i)][:])
        return new_state

    # scorer interface methods
    def init_state(self, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0))]
        z_list = [self.zero_state(x[0].unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)))
            z_list.append(self.zero_state(x[0].unsqueeze(0)))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att) - 1)
        if self.num_encs == 1:
            a = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han
        return dict(
            c_prev=c_list[:],
            z_prev=z_list[:],
            a_prev=a,
            workspace=(att_idx, z_list, c_list),
        )

    def score(self, yseq, state, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att[att_idx](
                x[0].unsqueeze(0),
                [x[0].size(0)],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"],
            )
        else:
            att_w = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs):
                att_c_list[idx], att_w[idx] = self.att[idx](
                    x[idx].unsqueeze(0),
                    [x[idx].size(0)],
                    self.dropout_dec[0](state["z_prev"][0]),
                    state["a_prev"][idx],
                )
            h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w[self.num_encs] = self.att[self.num_encs](
                h_han,
                [self.num_encs],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"][self.num_encs],
            )
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(
            ey, z_list, c_list, state["z_prev"], state["c_prev"]
        )
        if self.context_residual:
            logits = self.output(
                torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
            )
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return (
            logp,
            dict(
                c_prev=c_list[:],
                z_prev=z_list[:],
                a_prev=att_w,
                workspace=(att_idx, z_list, c_list),
            ),
        )


def decoder_for(args, odim, sos, eos, att, labeldist, meetingKB=None):
    return Decoder(
        args.eprojs,
        odim,
        args.dtype,
        args.dlayers,
        args.dunits,
        sos,
        eos,
        att,
        args.verbose,
        args.char_list,
        labeldist,
        args.lsm_weight,
        args.sampling_probability,
        args.dropout_rate_decoder,
        getattr(args, "context_residual", False),  # use getattr to keep compatibility
        getattr(args, "replace_sos", False),  # use getattr to keep compatibility
        getattr(args, "num_encs", 1),
        getattr(args, "wordemb", 0), # Starting KB-related args
        getattr(args, "lm_odim", 0),
        meetingKB,
        getattr(args, "KBlextree", False),
        getattr(args, "PtrGen", False),
        getattr(args, "PtrSche", 0),
        getattr(args, "PtrKBin", False),
        getattr(args, "smoothprob", 1.0),
        getattr(args, "attn_dim", args.dunits),
        getattr(args, "acousticonly", True),
        getattr(args, "additive_attn", False),
        getattr(args, "ooKBemb", False),
        getattr(args, "DBmask", 0.0),
        getattr(args, "DBinput", False),
        getattr(args, "treetype", ''),
        getattr(args, "treehid", 0),
        getattr(args, "boostfactor", 0),
        getattr(args, "stacked", False),
    )  # use getattr to keep compatibility
