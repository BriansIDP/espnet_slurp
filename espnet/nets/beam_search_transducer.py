"""Search algorithms for transducer models."""

from typing import List
from typing import Union
import os

import math
import time
import numpy as np
import torch
import editdistance

from espnet.nets.pytorch_backend.transducer.utils import create_lm_batch_state
from espnet.nets.pytorch_backend.transducer.utils import init_lm_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import recombine_hyps
from espnet.nets.pytorch_backend.transducer.utils import select_lm_state
from espnet.nets.pytorch_backend.transducer.utils import substract
from espnet.nets.transducer_decoder_interface import Hypothesis
from espnet.nets.transducer_decoder_interface import NSCHypothesis
from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface


class BeamSearchTransducer:
    """Beam search implementation for transducer."""

    def __init__(
        self,
        decoder: Union[TransducerDecoderInterface, torch.nn.Module],
        joint_network: torch.nn.Module,
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        score_norm: bool = True,
        nbest: int = 1,
        KBmodules: dict = {},
        char_list: list = [],
        sel_lm: torch.nn.Module = None,
        topk: int = 0
    ):
        """Initialize transducer beam search.

        Args:
            decoder: Decoder class to use
            joint_network: Joint Network class
            beam_size: Number of hypotheses kept during search
            lm: LM class to use
            lm_weight: lm weight for soft fusion
            search_type: type of algorithm to use for search
            max_sym_exp: number of maximum symbol expansions at each time step ("tsd")
            u_max: maximum output sequence length ("alsd")
            nstep: number of maximum expansion steps at each time step ("nsc")
            prefix_alpha: maximum prefix length in prefix search ("nsc")
            score_norm: normalize final scores by length ("default")
            nbest: number of returned final hypothesis
        """
        self.decoder = decoder
        self.joint_network = joint_network
        # Load KB-related modules
        self.meeting_KB = None
        self.DBinput = False
        self.PtrGen = False
        self.eos = decoder.odim - 1
        for name, module in KBmodules.items():
            setattr(self, name, module)

        self.beam_size = beam_size
        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim
        self.blank = decoder.blank

        self.search_type = search_type
        if self.beam_size <= 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            self.search_algorithm = self.nsc_beam_search
        elif search_type == "batched":
            self.search_algorithm = self.batched_beam_search_restricted
        else:
            raise NotImplementedError

        self.second_search_algorithm = self.default_beam_search_multiproc
        self.lm = lm
        self.lm_weight = lm_weight

        if lm is not None:
            self.use_lm = True
            self.is_wordlm = True if hasattr(lm.predictor, "wordlm") else False
            self.lm_predictor = lm.predictor.wordlm if self.is_wordlm else lm.predictor
            self.lm_layers = len(self.lm_predictor.rnn)
        else:
            self.use_lm = False

        self.max_sym_exp = max_sym_exp
        self.u_max = u_max
        self.nstep = nstep
        self.prefix_alpha = prefix_alpha
        self.score_norm = score_norm

        self.nbest = nbest
        self.char_list = getattr(self, 'char_list', char_list)
        self.sel_lm = sel_lm
        self.topk = topk

    def __call__(self, h: torch.Tensor, meetings=None, estlm=None, estlm_factor=0.0, prev_hid=None) -> Union[List[Hypothesis], List[NSCHypothesis]]:
        """Perform beam search.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(h.device)

        start = time.time()
        if not hasattr(self.decoder, "decoders"):
            self.decoder.set_data_type(h.dtype)

        if self.search_type == "default":
            nbest_hyps = self.search_algorithm(h, meetings=meetings,
                estlm=estlm, estlm_factor=estlm_factor, prev_hid=prev_hid)
        else:
            nbest_hyps = self.search_algorithm(h)
        print('Time elapsed: {:.2f}'.format(time.time()-start))
        print(nbest_hyps[0].score)
        nbest_hyps[0].lextree = None
        # print(meetings)

        return nbest_hyps

    def sort_nbest(
        self, hyps: Union[List[Hypothesis], List[NSCHypothesis]]
    ) -> Union[List[Hypothesis], List[NSCHypothesis]]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: list of hypotheses

        Return:
            hyps: sorted list of hypotheses

        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def greedy_search(self, h: torch.Tensor, meetings=None, estlm=None, estlm_factor=0.0) -> List[Hypothesis]:
        """Greedy search implementation for transformer-transducer.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            hyp: 1-best decoding results

        """
        dec_state = self.decoder.init_state(1)
        # Initialise prefix tree
        init_lextree = None
        if self.meeting_KB is not None:
            meeting_info = self.meeting_KB.get_meeting_KB(meetings, 1)
            if isinstance(meeting_info[2][0], str):
                init_lextree = self.meetingKB.meetinglextree[meeting_info[2][0]].copy()
            else:
                init_lextree = meeting_info[2][0]

        hyp = Hypothesis(score=0.0, vscore=h.new_zeros(1), yseq=[self.blank], dec_state=dec_state, lextree=init_lextree)
        cache = {}

        y, state, _ = self.decoder.score(hyp, cache)

        for i, hi in enumerate(h):
            # KB-trie search
            tree_track = None
            hs_plm = None
            if self.meeting_KB is not None:
                vy = hyp.yseq[-1] if len(hyp.yseq) > 1 else self.eos
                tree_track, lex_mask, inKB = self.meeting_lextree_step(vy, hyp.lextree, meeting_info[2][0])
                if self.PtrGen:
                    query_char = self.decoder.embed(torch.LongTensor([vy])).squeeze(0)
                    query_char = self.Qproj_char(query_char)
                    query_acoustic = self.Qproj_acoustic(hi)
                    query = query_char + query_acoustic
                    KBembedding, ptr_dist = self.get_KB_emb(query, lex_mask)
                    if self.KBin and not self.DBinput:
                        hs_plm = KBembedding

            z, joint_activation = self.joint_network(hi, y, h_plm=hs_plm)

            if self.meeting_KB is not None and self.PtrGen and inKB:
                p_gen = torch.sigmoid(self.pointer_gate(torch.cat([joint_activation, KBembedding], dim=-1)))
                p_gen = p_gen * self.smoothprob
                model_tu = torch.softmax(z, dim=-1)
                ptr_dist_fact = ptr_dist[1:] * (1.0 - model_tu[0])
                ptr_gen_complement = ptr_dist[-1:] * p_gen
                p_partial = ptr_dist_fact[:-1] * p_gen + model_tu[1:] * (1 - p_gen + ptr_gen_complement)
                ytu = torch.cat([model_tu[0:1], p_partial], dim=-1)
                ytu = torch.log(ytu)
            else:
                ytu = torch.log_softmax(z, dim=-1)
            logp, pred = torch.max(ytu, dim=-1)

            if pred != self.blank:
                # print(self.char_list[vy])
                # print(ptr_dist[:-1].sum() * p_gen)
                hyp.yseq.append(int(pred))
                hyp.score += float(logp)
                hyp.lextree = tree_track

                hyp.dec_state = state

                y, state, _ = self.decoder.score(hyp, cache)

        return [hyp]

    def meeting_lextree_step(self, char_idx, new_tree, meeting):
        step_mask = torch.ones(len(self.char_list) + 1)
        new_tree = new_tree[0]
        char_idx = char_idx if isinstance(char_idx, int) else char_idx.item()
        ptr_gen = True
        if char_idx in [self.eos] or self.char_list[char_idx].endswith('▁'):
            new_tree = self.meeting_KB.meetinglextree[meeting] if isinstance(meeting, str) else meeting
            ptr_gen = True
        elif char_idx not in new_tree:
            new_tree = [{}]
            ptr_gen = False
        else:
            new_tree = new_tree[char_idx]
        step_mask[list(new_tree[0].keys())] = 0
        step_mask[-1] = 0
        # step_mask[0] = 0
        return new_tree, step_mask.byte(), ptr_gen

    def get_KB_emb(self, query, mask):
        # meeting_KB = torch.cat([self.joint_network.lin_out.weight, self.ooKBemb.weight], dim=0)
        meeting_KB = torch.cat([self.decoder.embed.weight, self.ooKBemb.weight], dim=0)
        meeting_KB = self.Kproj(meeting_KB)
        # attn_dim, nbpe * attn_dim -> nbpe
        KBweight = torch.einsum('j,ij->i', query, meeting_KB)
        KBweight = KBweight / math.sqrt(query.size(-1))
        KBweight.masked_fill_(mask.bool(), -1e9)
        KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
        # nbpe, nbpe * attn_dim -> attn_dim
        KBembedding = torch.einsum('i,ij->j', KBweight[:-1], meeting_KB[:-1,:])
        # KBembedding = torch.einsum('i,ij->j', KBweight, meeting_KB)
        return KBembedding, KBweight

    def get_KB_emb_batch(self, query, mask):
        # meeting_KB = torch.cat([self.joint_network.lin_out.weight, self.ooKBemb.weight], dim=0)
        meeting_KB = torch.cat([self.decoder.embed.weight, self.ooKBemb.weight], dim=0)
        meeting_KB = self.Kproj(meeting_KB)
        # bb * attn_dim, nbpe * attn_dim -> bb * nbpe
        KBweight = torch.einsum('kj,ij->ki', query, meeting_KB)
        KBweight = KBweight / math.sqrt(query.size(-1))
        KBweight.masked_fill_(mask.bool(), -1e9)
        KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
        # bb * nbpe, nbpe * attn_dim -> bb * attn_dim
        KBembedding = torch.einsum('ki,ij->kj', KBweight[:,:-1], meeting_KB[:-1,:])
        # KBembedding = torch.einsum('ki,ij->j', KBweight, meeting_KB)
        return KBembedding, KBweight

    def get_meetingKB_emb_map(self, query, meeting_mask, meeting_embs, back_transform):
        """
        query: (nutts, T, attn_dim)
        meeting_embs: (nutts, n_nodes, embdim)
        meeting_mask: (nutts, n_nodes)
        back_transform: (nutts, n_nodes, nbpes)

        return: (nutts, T, nbpes)
        """
        nutts = meeting_embs.size(0)
        n_nodes = meeting_embs.size(1)
        T = query.size(1)
        if self.gnnheads > 1 and not self.jknet:
            # meeting_KB = meeting_embs.view(nutts, n_nodes, self.gnnheads, self.tree_hid)
            meeting_KB = self.Kproj(meeting_embs.view(nutts, n_nodes, self.gnnheads, -1))
            query = query.view(nutts, T, self.gnnheads, -1)
            meeting_mask = meeting_mask.bool().view(nutts, 1, n_nodes, 1)
            KBweight = torch.einsum('ijnk,itnk->itjn', meeting_KB, query) / math.sqrt(query.size(-1))
            KBweight.masked_fill_(meeting_mask, -1e9)
            KBweight = KBweight.contiguous() #.view(nutts, T, -1)
            KBweight = torch.nn.functional.softmax(KBweight, dim=2) #.view(nutts, T, n_nodes, -1)
            if meeting_embs.size(1) > 1:
                KBembedding = torch.einsum('ijnk,itjn->itnk', meeting_KB[:,:-1,:,:], KBweight[:,:,:-1,:])
                # KBembedding = KBembedding.view(nutts, T, -1)
                headweights = torch.tensor([0.8, 0.2]).to(KBembedding.device).unsqueeze(0).unsqueeze(0).repeat(nutts, T, 1)
                # headweights = (KBembedding * query).sum(dim=-1) / math.sqrt(query.size(-1)) # itnk,itnk->itn
                # headweights = torch.softmax(headweights, dim=-1)
                KBweight = torch.einsum('itjn,itn->itj', KBweight, headweights)
                KBembedding = torch.einsum('itnk,itn->itk', KBembedding, headweights)
            else:
                KBembedding = KBweight.new_zeros(nutts, T, meeting_KB.size(-1))
                KBweight = KBweight.mean(dim=-1)
        elif  self.jknet and self.gnnheads > 1:
            meeting_embs = meeting_embs.view(nutts, n_nodes, self.gnnheads, self.tree_hid)
            # stitch_keys = self.stitchproj(meeting_embs)
            # stitch_weights = torch.einsum('itk,ijnk->itjn', query, stitch_keys) / math.sqrt(query.size(-1))
            # stitch_weights = torch.softmax(stitch_weights, dim=-1)
            stitch_weights = torch.tensor([0.8, 0.2]).to(
                meeting_embs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(nutts, T, n_nodes, 1)
            meeting_KB = self.Kproj(meeting_embs)
            KBweight = torch.einsum('ijnk,itk->itjn', meeting_KB, query)
            KBweight = (KBweight * stitch_weights).sum(dim=-1)
            KBweight = KBweight / math.sqrt(query.size(-1))
            KBweight.masked_fill_(meeting_mask.bool().unsqueeze(1).repeat(1, query.size(1), 1), -1e9)
            KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
            if meeting_embs.size(1) > 1:
                finegrained_weights = stitch_weights * KBweight.unsqueeze(-1)
                KBembedding = torch.einsum(
                    'ijnk,itjn->itk', meeting_KB[:,:-1,:,:], finegrained_weights[:,:,:-1,:])
            else:
                KBembedding = KBweight.new_zeros(meeting_KB.size(0), query.size(1), meeting_KB.size(-1))
        else:
            meeting_KB = self.Kproj(meeting_embs)
            KBweight = torch.einsum('ijk,itk->itj', meeting_KB, query)
            KBweight = KBweight / math.sqrt(query.size(-1))
            KBweight.masked_fill_(meeting_mask.bool().unsqueeze(1).repeat(1, query.size(1), 1), -1e9)
            KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
            if meeting_embs.size(1) > 1:
                KBembedding = torch.einsum('ijk,itj->itk', meeting_KB[:,:-1,:], KBweight[:,:,:-1])
            else:
                KBembedding = KBweight.new_zeros(meeting_KB.size(0), query.size(1), meeting_KB.size(-1))
        KBweight = torch.einsum('ijk,itj->itk', back_transform, KBweight)
        return KBembedding, KBweight

    def get_lextree_step_embs_inference(self, char_idx, tree_track, reset_lextree):
        # meeting_KB = torch.cat([self.embed.weight.data, self.ooKBemb.weight], dim=0)
        ooKB_id = len(self.char_list)
        new_tree = tree_track[0]
        char_idx = char_idx if isinstance(char_idx, int) else char_idx.item()
        ptr_gen = True
        if char_idx in [self.eos] or self.char_list[char_idx].endswith('▁'):
            new_tree = reset_lextree
        elif char_idx not in new_tree:
            new_tree = [{}]
            ptr_gen = False
        else:
            new_tree = new_tree[char_idx]
        if len(new_tree) > 2 and new_tree[0] != {}:
            step_embs = torch.cat([node[3] for key, node in new_tree[0].items()], dim=0)
        else:
            step_embs = torch.empty(0, self.tree_hid * self.gnnheads)
        back_transform = []
        indices = list(new_tree[0].keys()) + [ooKB_id]
        for i, ind in enumerate(indices):
            one_hot = [0] * (ooKB_id + 1)
            one_hot[ind] = 1
            back_transform.append(one_hot)
        back_transform = torch.Tensor(back_transform)
        # step_embs = torch.einsum('jk,km->jm', back_transform, meeting_KB)
        step_embs = torch.cat([step_embs, self.ooKBemb.weight], dim=0)
        step_mask = torch.zeros(back_transform.size(0)).byte()
        return step_mask.unsqueeze(0), new_tree, ptr_gen, step_embs.unsqueeze(0), back_transform.unsqueeze(0)

    def get_lextree_encs(self, lextree, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            ey = self.decoder.embed(torch.LongTensor([wordpiece]))
            lextree.append(ey)
            return ey
        elif lextree[1] == -1 and lextree[0] != {}:
            wordpieces = []
            for newpiece, values in lextree[0].items():
                wordpieces.append(self.get_lextree_encs(values, newpiece))
            wordpiece_h = torch.cat(wordpieces, dim=0)
            if self.treetype.endswith('mean'):
                wordpiece_h = wordpiece_h.mean(dim=0).unsqueeze(0)
            elif self.treetype.endswith('max'):
                wordpiece_h = wordpiece_h.max(dim=0)[0].unsqueeze(0)
            else:
                wordpiece_h = wordpiece_h.sum(dim=0).unsqueeze(0)
            if wordpiece is not None:
                ey = self.decoder.embed(torch.LongTensor([wordpiece]))
                wordpiece_h = self.recursive_proj(torch.cat([ey, wordpiece_h], dim=-1))
                wordpiece_h = torch.relu(wordpiece_h)
                # wordpiece_h = torch.sigmoid(wordpiece_h)
                lextree.append(wordpiece_h)
            return wordpiece_h

    def get_last_word(self, charlist, split=False):
        starts = len(charlist) - 2
        char_tuple = [self.char_list[charlist[-1]]] if not split else [charlist[-1]]
        while starts > 0 and not self.char_list[charlist[starts]].endswith('▁'):
            char_tuple = [self.char_list[charlist[starts]] if not split else charlist[starts]] + char_tuple
            starts -= 1
        return tuple(char_tuple)

    def batched_beam_search_restricted(self, hs: torch.Tensor, h_mask: torch.Tensor, meetings=None) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            hs: Encoded speech features (N_batch, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        dec_state = self.decoder.init_state(1)

        # TCPGen
        init_lextree = None
        if self.meeting_KB is not None and meetings is not None:
            if isinstance(meetings[2][0], str):
                init_lextree = self.meeting_KB.meetinglextree[meetings[2][0]].copy()
            else:
                init_lextree = meetings[2][0]

        kept_hyps = [[Hypothesis(score=0.0, vscore=hs.new_zeros(1), yseq=[self.blank], dec_state=dec_state, lextree=init_lextree, p_gen=[])]
            for i in range(hs.size(0))]
        max_steps = hs.size(1)
        for i in range(max_steps):
            beam_v = beam if i > 0 else 1
            unfinished_inds = [ind for ind, j in enumerate(h_mask) if j > i]
            beam_batch_enc = []
            beam_batch_dec_h = []
            beam_batch_dec_c = []
            beam_batch_ys = []
            bb_vscores = []
            batch = len(unfinished_inds)
            for j in unfinished_inds:
                hyps = kept_hyps[j]
                beam_batch_enc.append(hs[j:j+1, i].repeat(len(hyps), 1))
                beam_batch_dec_h += [hyp.dec_state[0] for hyp in hyps]
                beam_batch_dec_c += [hyp.dec_state[1] for hyp in hyps]
                beam_batch_ys += [hyp.yseq[-1] for hyp in hyps]
                bb_vscores += [hyp.vscore for hyp in hyps]
            unfinished_kept_hyps = []
            for j, hyp in enumerate(kept_hyps):
                unfinished_kept_hyps.append(hyp if j not in unfinished_inds else [])
            # bb * dec_dim
            beam_batch_dec = (torch.cat(beam_batch_dec_h, dim=1), torch.cat(beam_batch_dec_c, dim=1))
            # bb * enc_dim
            beam_batch_enc = torch.cat(beam_batch_enc, dim=0)
            # bb * 1
            beam_batch_ys = torch.LongTensor(beam_batch_ys).to(hs.device).view(-1, 1)
            # bb * 1
            bb_vscores = torch.cat(bb_vscores, dim=0)

            # forward one step
            # bb * 1 * dec_dim
            dec_output, new_states = self.decoder.rnn_forward_batch(beam_batch_ys, beam_batch_dec)

            # TCPGen
            hs_plm = None
            step_mask = torch.ones(batch, beam_v, len(self.char_list) + 1)
            inKB_mat = torch.ones(batch, beam_v)
            if self.meeting_KB is not None and meetings is not None and self.PtrGen:
                for b_id, j in enumerate(unfinished_inds):
                    hyps = kept_hyps[j]
                    for hyp_id, hyp in enumerate(hyps):
                        step_mask[b_id, hyp_id, list(hyp.lextree[0].keys())] = 0
                        inKB_mat[b_id, hyp_id] = 0 if len(hyp.lextree[0].keys()) else 1
                step_mask[:,:,-1] = 0
                query_char = self.decoder.embed(beam_batch_ys).squeeze(1)
                query_char = self.Qproj_char(query_char)
                query_acoustic = self.Qproj_acoustic(beam_batch_enc)
                # bb * attn_dim
                query = query_char + query_acoustic
                # bb * attn_dim, bb * odim
                KBembedding, ptr_dist = self.get_KB_emb_batch(query,
                    step_mask.view(-1, step_mask.size(-1)).to(query.device))
                if self.KBin:
                    hs_plm = self.dropout_KB(KBembedding)

            zs, joint_activations = self.joint_network(beam_batch_enc, dec_output.squeeze(1), h_plm=hs_plm)

            # TCPGen
            if self.meeting_KB is not None and meetings is not None and self.PtrGen:
                model_tu = torch.softmax(zs, dim=-1)
                p_gen = torch.sigmoid(self.pointer_gate(torch.cat([joint_activations, KBembedding], dim=-1)))
                p_gen = p_gen.masked_fill(inKB_mat.view(-1, 1).to(p_gen.device).bool(), 0) * self.smoothprob
                ptr_dist_fact = ptr_dist[:,1:] * (1.0 - model_tu[:,0:1])
                ptr_gen_complement = ptr_dist[:,-1:] * p_gen
                p_partial = ptr_dist_fact[:,:-1] * p_gen + model_tu[:,1:] * (1 - p_gen + ptr_gen_complement)
                bb_ytu = torch.cat([model_tu[:,0:1], p_partial], dim=-1)
                bb_ytu = torch.log(bb_ytu).view(batch, beam_v, -1)
            else:
                # batch * beam * odim
                bb_ytu = torch.log_softmax(zs, dim=-1).view(batch, beam_v, -1)

            # batch * beam * odim
            bb_vscores = bb_vscores.view(batch, beam_v, 1).repeat(1, 1, bb_ytu.size(-1))
            bb_vscores = bb_vscores + bb_ytu
            bb_vscores_null = bb_vscores[:,:,0:1].contiguous().view(batch, -1) # batch * beam
            bb_vscores_other = bb_vscores[:,:,1:].contiguous().view(batch, -1) # batch * (beam * odim-1)

            # get top k word piece outputs
            accum_best_scores, accum_best_ids = torch.topk(bb_vscores_other, beam, 1)
            # batch * beam
            accum_odim_ids = torch.fmod(accum_best_ids, bb_ytu.size(-1)-1).view(-1, 1) + 1 # starting from 1
            pad_b = (torch.arange(batch) * beam_v).view(-1, 1).to(accum_best_ids.device)
            # batch * beam
            accum_beam_ids = torch.div(accum_best_ids, bb_ytu.size(-1)-1, rounding_mode='floor')
            accum_padded_beam_ids = (accum_beam_ids + pad_b).view(-1)

            # gather yseq and forward again
            new_states_h = new_states[0][:, accum_padded_beam_ids]
            new_states_c = new_states[1][:, accum_padded_beam_ids]
            new_dec_output, _ = self.decoder.rnn_forward_batch(accum_odim_ids, (new_states_h, new_states_c))
            if i == 0:
                beam_batch_enc = beam_batch_enc.unsqueeze(1).repeat(1, beam, 1).view(new_dec_output.size(0), -1)
            new_zs, _ = self.joint_network(beam_batch_enc, new_dec_output.squeeze(1))
            bb_ytu = torch.log_softmax(new_zs, dim=-1)[:, 0:1].view(batch, -1)
            accum_best_scores = accum_best_scores + bb_ytu # batch * beam

            # combine these with null-outputing sequences and find topk
            bb_vscores_candidates = torch.cat([bb_vscores_null, accum_best_scores], dim=-1)
            bb_vscores, bb_inds = torch.topk(bb_vscores_candidates, beam, dim=1)
            bb_inds = bb_inds.data.cpu().tolist()

            for batch_id, indices in enumerate(bb_inds):
                for ind_id, index in enumerate(indices):
                    new_vscore = bb_vscores[batch_id, ind_id:ind_id+1]
                    if index >= beam_v:
                        beam_id = accum_beam_ids[batch_id, index - beam_v]
                        batchbeam_index = (index - beam_v) + batch_id * beam
                        new_vy = accum_odim_ids[batchbeam_index].item()
                        new_yseq = kept_hyps[batch_id][beam_id].yseq[:] + [new_vy]
                        new_lextree = None
                        if self.meeting_KB is not None:
                            new_lextree, _, _ = self.meeting_lextree_step(new_vy,
                                kept_hyps[batch_id][beam_id].lextree, init_lextree) 
                        new_vscore = bb_vscores[batch_id, ind_id:ind_id+1]
                        # search and delete duplicate
                        for ind, hyp in enumerate(unfinished_kept_hyps[unfinished_inds[batch_id]]):
                            if new_yseq == hyp.yseq and new_vscore.item() < hyp.score:
                                new_vscore += -1e9
                        unfinished_kept_hyps[unfinished_inds[batch_id]].append(
                            Hypothesis(
                                score=new_vscore.item(),
                                vscore=new_vscore,
                                yseq=new_yseq,
                                dec_state=(new_states_h[:, batchbeam_index].unsqueeze(1), new_states_c[:, batchbeam_index].unsqueeze(1)),
                                lextree=new_lextree
                            )
                        )
                    else:
                        if i == max_steps - 1:
                            for ind, hyp in enumerate(unfinished_kept_hyps[unfinished_inds[batch_id]]):
                                if kept_hyps[batch_id][index].yseq == hyp.yseq and new_vscore.item() < hyp.score:
                                    new_vscore += -1e9
                        if i != max_steps - 1 or new_vscore > -10000:
                            unfinished_kept_hyps[unfinished_inds[batch_id]].append(
                                Hypothesis(
                                    score=new_vscore.item(),
                                    vscore=new_vscore,
                                    yseq=kept_hyps[batch_id][index].yseq,
                                    dec_state=kept_hyps[batch_id][index].dec_state,
                                    lextree=kept_hyps[batch_id][index].lextree
                                )
                            )
            kept_hyps = unfinished_kept_hyps
        return kept_hyps

    def batched_beam_search(self, hs: torch.Tensor, h_mask: torch.Tensor, meetings=None) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            hs: Encoded speech features (N_batch, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        batch = len(hs)
        dec_state = self.decoder.init_state(1)
        vscores = torch.zeros(batch, beam).to(hs.device)

        kept_hyps = [[Hypothesis(score=0.0, vscore=hs.new_zeros(1), yseq=[self.blank], dec_state=dec_state)] for i in range(batch)]
        max_steps = hs.size(1)
        for i in range(max_steps):
            unfinished_inds = [ind for ind, j in enumerate(h_mask) if j > i]
            batched_hyps = [kept_hyps[j] for j in unfinished_inds] # batch * beam
            unfinished_h = hs[torch.LongTensor(unfinished_inds), i]
            unfinished_kept_hyps = [[] for j in range(len(unfinished_inds))]
            done_list = [1] * len(unfinished_inds)

            while True:
                max_hyp_batch = []
                do_list = [ind for ind, value in enumerate(done_list) if value == 1]
                for j in do_list:
                    hyps = batched_hyps[j]
                    best_hyp = max(hyps, key=lambda x: x.score)
                    max_hyp_batch.append(best_hyp)
                    hyps.remove(best_hyp)
                    batched_hyps[j] = hyps if hyps else []
                y_history = torch.LongTensor([hyp.yseq[-1] for hyp in max_hyp_batch]).to(hs.device)
                dec_states_h = torch.cat([hyp.dec_state[0] for hyp in max_hyp_batch], dim=1)
                dec_states_c = torch.cat([hyp.dec_state[1] for hyp in max_hyp_batch], dim=1)
                # batch * D_dec
                dec_output, new_states = self.decoder.rnn_forward_batch(y_history.view(-1, 1),
                    (dec_states_h, dec_states_c))

                zs, joint_activations = self.joint_network(unfinished_h[do_list], dec_output.squeeze(1))
                ytu_batch = torch.log_softmax(zs, dim=-1)
                top_k_value, top_k_inds = ytu_batch[:, 1:].topk(beam, dim=-1)
                # for j, max_hyp in enumerate(max_hyp_batch):
                #     if done_list[j] == 1:
                for ind, j in enumerate(do_list):
                    max_hyp = max_hyp_batch[ind]
                    position = -1
                    for subind, hyp in enumerate(unfinished_kept_hyps[j]):
                        if hyp.yseq == max_hyp.yseq and hyp.score < (max_hyp.score + float(ytu_batch[ind, 0:1])):
                            position = subind
                        elif hyp.yseq == max_hyp.yseq:
                            position = -2
                    if position >= 0:
                        unfinished_kept_hyps[j][position] = Hypothesis(
                            score=(max_hyp.score + float(ytu_batch[ind, 0:1])),
                            vscore=max_hyp.vscore + ytu_batch[ind, 0:1],
                            yseq=max_hyp.yseq[:],
                            dec_state=max_hyp.dec_state,
                        )
                    elif position == -1:
                        unfinished_kept_hyps[j].append(
                            Hypothesis(
                                score=(max_hyp.score + float(ytu_batch[ind, 0:1])),
                                vscore=max_hyp.vscore + ytu_batch[ind, 0:1],
                                yseq=max_hyp.yseq[:],
                                dec_state=max_hyp.dec_state,
                            )
                        )
                    # unfinished_kept_hyps[j].append(
                    #     Hypothesis(
                    #         score=(max_hyp.score + float(ytu_batch[ind, 0:1])),
                    #         vscore=max_hyp.vscore + ytu_batch[ind, 0:1],
                    #         yseq=max_hyp.yseq[:],
                    #         dec_state=max_hyp.dec_state,
                    #     )
                    # )
                    for logp, k in zip(*(top_k_value[ind], top_k_inds[ind])):
                        batched_hyps[j].append(
                            Hypothesis(
                                score=max_hyp.score + float(logp),
                                vscore=max_hyp.vscore + logp,
                                yseq=max_hyp.yseq[:] + [int(k + 1)],
                                dec_state=(new_states[0][:, ind:ind+1], new_states[1][:, ind:ind+1]),
                            )
                        )

                    best_hyps = max(batched_hyps[j], key=lambda x: x.score)
                    hyps_max = float(best_hyps.score)
                    max_yseq = best_hyps.yseq
                    kept_most_prob = sorted(
                        [hyp for hyp in unfinished_kept_hyps[j] if hyp.score > hyps_max],
                        key=lambda x: x.score,
                    )
                    if len(kept_most_prob) >= beam or (len(max_yseq) >= max_steps*1.5 and len(kept_most_prob) > 0):
                        unfinished_kept_hyps[j] = kept_most_prob
                        done_list[j] = 0

                if sum(done_list) == 0:
                    for ind, j in enumerate(unfinished_inds):
                        kept_hyps[j] = unfinished_kept_hyps[ind]
                    break

        return kept_hyps

    def default_beam_search_multiproc(self, h: torch.Tensor, y_seq, meetings=None) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        # print('parent process:', os.getppid())
        # print('process id:', os.getpid())
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        init_lextree = None
        if self.meeting_KB is not None:
            meeting_info = self.meeting_KB.get_meeting_KB(meetings, 1)
            if isinstance(meeting_info[2][0], str):
                init_lextree = self.meeting_KB.meetinglextree[meeting_info[2][0]].copy()
            else:
                init_lextree = meeting_info[2][0]

        kept_hyps = [Hypothesis(score=0.0, vscore=h.new_zeros(1), yseq=[self.blank], dec_state=dec_state, lextree=init_lextree, p_gen=[])]
        cache = {}

        seq_true = [self.char_list[int(idx)] for idx in y_seq if int(idx) != -1]
        seq_true_text = "".join(seq_true).replace('▁', ' ')
        ref_words = seq_true_text.split()

        for enc_pos, hi in enumerate(h):
            hyps = kept_hyps
            # import pdb; pdb.set_trace()
            kept_hyps = []
            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                y, state, lm_tokens = self.decoder.score(max_hyp, cache)
                if lm_tokens.item() == 0:
                    lm_tokens = lm_tokens + self.eos - 1
                else:
                    lm_tokens = lm_tokens - 1

                # KB-trie search
                tree_track = None
                hs_plm = None
                if self.meeting_KB is not None:
                    vy = max_hyp.yseq[-1] if len(max_hyp.yseq) > 1 else self.eos
                    tree_track, lex_mask, inKB = self.meeting_lextree_step(vy, max_hyp.lextree, meeting_info[2][0])
                    if self.DBinput:
                        hs_plm = self.DBembed((1-lex_mask[:-1].float()))

                    if self.PtrGen and inKB:
                        query_char = self.decoder.embed(torch.LongTensor([vy])).squeeze(0)
                        query_char = self.Qproj_char(query_char)
                        query_acoustic = self.Qproj_acoustic(hi)
                        query = query_char + query_acoustic
                        KBembedding, ptr_dist = self.get_KB_emb(query, lex_mask)
                        if self.KBin and not self.DBinput:
                            hs_plm = KBembedding

                z, joint_activation = self.joint_network(hi, y, h_plm=hs_plm)
                model_tu = torch.softmax(z, dim=-1)
                store_pgen = 0

                if self.meeting_KB is not None and self.PtrGen and inKB and model_tu[0] < 1.0:
                    p_gen = torch.sigmoid(self.pointer_gate(torch.cat([joint_activation, KBembedding], dim=-1)))
                    p_gen = p_gen * self.smoothprob
                    # Apply factorised distribution
                    ptr_dist_fact = ptr_dist[1:] * (1.0 - model_tu[0])
                    ptr_gen_complement = ptr_dist[-1:] * p_gen
                    p_partial = ptr_dist_fact[:-1] * p_gen + model_tu[1:] * (1 - p_gen + ptr_gen_complement)
                    ytu = torch.cat([model_tu[0:1], p_partial], dim=-1)
                    store_pgen = round((ptr_dist_fact[1:-1].sum() * p_gen).item(), 2)
                    ytu = torch.log(ytu)
                else:
                    ytu = torch.log(model_tu)
                    # ytu = torch.log_softmax(z, dim=-1)
                top_k = ytu[1:].topk(beam_k, dim=-1)

                position = -1
                for ind, hyp in enumerate(kept_hyps):
                    if hyp.yseq == max_hyp.yseq and hyp.score < (max_hyp.score + float(ytu[0:1])):
                        position = ind
                    elif hyp.yseq == max_hyp.yseq:
                        position = -2
                if position >= 0:
                    kept_hyps[ind] = Hypothesis(
                        score=(max_hyp.score + float(ytu[0:1])),
                        vscore=0, #max_hyp.vscore + ytu[0:1],
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                        lextree=max_hyp.lextree,
                        p_gen=max_hyp.p_gen,
                        )
                elif position == -1:
                    kept_hyps.append(
                        Hypothesis(
                            score=(max_hyp.score + float(ytu[0:1])),
                            vscore=0, #max_hyp.vscore + ytu[0:1],
                            yseq=max_hyp.yseq[:],
                            dec_state=max_hyp.dec_state,
                            lm_state=max_hyp.lm_state,
                            lextree=max_hyp.lextree,
                            p_gen=max_hyp.p_gen,
                        )
                    )


                if self.use_lm:
                    lm_state, lm_scores = self.lm.predict(max_hyp.lm_state, lm_tokens)
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        # LM removed blank when being loaded
                        score += self.lm_weight * lm_scores[0][k].item()

                    hyps.append(
                        Hypothesis(
                            score=score,
                            vscore=0, # max_hyp.vscore + logp,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                            lextree=tree_track,
                            p_gen=max_hyp.p_gen + [store_pgen],
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                max_yseq = max(hyps, key=lambda x: x.score).yseq
                # kept_hyps_max = sorted([x.score for x in kept_hyps])[-2:]
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                # print(enc_pos, hyps_max, kept_hyps_max)
                if len(kept_most_prob) >= beam or (len(max_yseq) >= len(h)*1.5 and len(kept_most_prob) > 0):
                    kept_hyps = kept_most_prob
                    break
        to_return = []
        for hyp in kept_hyps:
            seq_hat = [self.char_list[int(idx)] for idx in hyp.yseq if int(idx) != 0]
            seq_hat_text = seq_hat_text = "".join(seq_hat).replace('▁', ' ')
            hyp_words = seq_hat_text.split()
            werror = editdistance.eval(hyp_words, ref_words)
            to_return.append((hyp.yseq[1:], hyp.score, werror))
        return to_return

    def default_beam_search(self, h: torch.Tensor, meetings=None, estlm=None, estlm_factor=0.0, prev_hid=None) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        init_lextree = None
        if self.meeting_KB is not None:
            if self.sel_lm is None:
                meeting_info = self.meeting_KB.get_meeting_KB(meetings, 1)
            else:
                meeting_info = (self.meeting_KB.full_wordlist, None, self.meeting_KB.full_lextree)

            if isinstance(meeting_info[2][0], str):
                init_lextree = self.meeting_KB.meetinglextree[meeting_info[2][0]].copy()
            else:
                init_lextree = meeting_info[2][0]
            init_hid = None
            new_hid = None
            if self.sel_lm is not None:
                init_hid = self.sel_lm.init_hidden(1) if prev_hid is None else prev_hid
            else:
                if self.treetype.startswith('gcn'):
                    self.gcn(init_lextree, self.decoder.embed)
                elif self.treetype.startswith('sage'):
                    self.sage(init_lextree, self.decoder.embed)
                elif self.treetype.startswith('appnp'):
                    self.appnp(init_lextree, self.decoder.embed)
                elif self.treetype.startswith('iigcn'):
                    self.gcnii(init_lextree, self.decoder.embed)
                elif self.treetype.startswith('comb'):
                    self.combined(init_lextree, self.decoder.embed)
                else:
                    self.get_lextree_encs(init_lextree)

        estlm_state = None
        if estlm is not None:
            estlm_state = estlm.init_hidden(1)

        kept_hyps = [Hypothesis(score=0.0, vscore=h.new_zeros(1), yseq=[self.blank], dec_state=dec_state,
            lextree=init_lextree, p_gen=[], trace=[], estlm_state=estlm_state)]
        cache = {}

        for enc_pos, hi in enumerate(h):
            hyps = kept_hyps
            # import pdb; pdb.set_trace()
            kept_hyps = []
            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                y, state, lm_tokens = self.decoder.score(max_hyp, cache)
                if lm_tokens.item() == 0:
                    lm_tokens = lm_tokens + self.eos - 1
                else:
                    lm_tokens = lm_tokens - 1

                # KB-trie search
                tree_track = None
                hs_plm = None
                if self.meeting_KB is not None:
                    vy = max_hyp.yseq[-1] if len(max_hyp.yseq) > 1 else self.eos
                    # select KB entries and organise reset tree
                    if self.sel_lm is not None and (vy == self.eos or self.char_list[vy].endswith('▁')):
                        if vy == self.eos:
                            thisword = self.meeting_KB.vocab.get_idx('<eos>')
                        else:
                            thisword = self.meeting_KB.vocab.get_idx(self.get_last_word(max_hyp.yseq))
                        lmout, new_hid, TPout = self.sel_lm(torch.LongTensor([[thisword]]), max_hyp.prev_hid)
                        # lmout = torch.log_softmax(lmout.squeeze(0).squeeze(0), dim=-1) - unigram_p * self.meetingKB.unigram_dist
                        lmout = lmout.squeeze(0).squeeze(0)
                        new_lm_output = lmout[meeting_info[0][0]]
                        values, new_select = torch.topk(new_lm_output, self.topk, dim=0)
                        reset_lextree = self.meeting_KB.get_tree_from_inds(new_select, extra=meeting_info[1])
                        self.get_lextree_encs(reset_lextree)
                    else:
                        meeting = meeting_info[2][0]
                        reset_lextree = self.meeting_KB.meetinglextree[meeting] if isinstance(meeting, str) else init_lextree
                        if self.sel_lm is not None:
                            new_hid = max_hyp.prev_hid

                    # tree_track, lex_mask, inKB = self.meeting_lextree_step(vy, max_hyp.lextree, meeting_info[2][0])
                    step_mask, tree_track, inKB, step_embs, back_transform = self.get_lextree_step_embs_inference(
                        vy, max_hyp.lextree, reset_lextree)
                    if self.DBinput:
                        hs_plm = self.DBembed((1-lex_mask[:-1].float()))

                    if self.PtrGen and inKB:
                        query_char = self.decoder.embed(torch.LongTensor([vy])).squeeze(0)
                        query_char = self.Qproj_char(query_char)
                        query_acoustic = self.Qproj_acoustic(hi)
                        query = (query_char + query_acoustic).unsqueeze(0).unsqueeze(0)
                        # KBembedding, ptr_dist = self.get_KB_emb(query, lex_mask)
                        KBembedding, ptr_dist = self.get_meetingKB_emb_map(query, step_mask, step_embs, back_transform)
                        KBembedding, ptr_dist = KBembedding.squeeze(0).squeeze(0), ptr_dist.squeeze(0).squeeze(0)
                        if self.KBin and not self.DBinput:
                            hs_plm = KBembedding.squeeze(0).squeeze(0)

                z, joint_activation = self.joint_network(hi, y, h_plm=hs_plm)
                model_tu = torch.softmax(z, dim=-1)
                store_pgen = 0

                if self.meeting_KB is not None and self.PtrGen and inKB and model_tu[0] < 1.0:
                    p_gen = torch.sigmoid(self.pointer_gate(torch.cat([joint_activation, KBembedding], dim=-1)))
                    p_gen = p_gen * self.smoothprob
                    # Apply factorised distribution
                    ptr_dist_fact = ptr_dist[1:] * (1.0 - model_tu[0])
                    ptr_gen_complement = ptr_dist[-1:] * p_gen
                    p_partial = ptr_dist_fact[:-1] * p_gen + model_tu[1:] * (1 - p_gen + ptr_gen_complement)
                    ytu = torch.cat([model_tu[0:1], p_partial], dim=-1)
                    # print(self.char_list[vy])
                    # print(ptr_dist[1:-1].sum() * p_gen)
                    # if len(max_hyp.yseq) > 1 and self.char_list[max_hyp.yseq[-2]] == 'TUR':
                    #     import pdb; pdb.set_trace()
                    # Apply normal distribution
                    # ptr_gen_complement = ptr_dist[-1:] * p_gen
                    # ytu = ptr_dist[:-1] * p_gen + model_tu * (1 - p_gen + ptr_gen_complement)
                    # store_pgen = round((ptr_dist_fact[1:-1].sum() * p_gen).item(), 2)
                    ytu = torch.log(ytu)
                else:
                    ytu = torch.log(model_tu)
                    # ytu = torch.log_softmax(z, dim=-1)
                top_k = ytu[1:].topk(beam_k, dim=-1)

                position = -1
                for ind, hyp in enumerate(kept_hyps):
                    if hyp.yseq == max_hyp.yseq and hyp.score < (max_hyp.score + float(ytu[0:1])):
                        position = ind
                    elif hyp.yseq == max_hyp.yseq:
                        position = -2
                if position >= 0:
                    kept_hyps[ind] = Hypothesis(
                            score=(max_hyp.score + float(ytu[0:1])),
                            vscore=max_hyp.vscore + ytu[0:1],
                            yseq=max_hyp.yseq[:],
                            dec_state=max_hyp.dec_state,
                            lm_state=max_hyp.lm_state,
                            lextree=max_hyp.lextree,
                            p_gen=max_hyp.p_gen,
                            trace=max_hyp.trace + [enc_pos],
                            estlm_state=max_hyp.estlm_state,
                            prev_hid=max_hyp.prev_hid
                        )
                elif position == -1:
                    kept_hyps.append(
                        Hypothesis(
                            score=(max_hyp.score + float(ytu[0:1])),
                            vscore=max_hyp.vscore + ytu[0:1],
                            yseq=max_hyp.yseq[:],
                            dec_state=max_hyp.dec_state,
                            lm_state=max_hyp.lm_state,
                            lextree=max_hyp.lextree,
                            p_gen=max_hyp.p_gen,
                            trace=max_hyp.trace + [enc_pos],
                            estlm_state=max_hyp.estlm_state,
                            prev_hid=max_hyp.prev_hid
                        )
                    )

                # kept_hyps.append(
                #     Hypothesis(
                #         score=(max_hyp.score + float(ytu[0:1])),
                #         vscore=0, # max_hyp.vscore + ytu[0:1],
                #         yseq=max_hyp.yseq[:],
                #         dec_state=max_hyp.dec_state,
                #         lm_state=max_hyp.lm_state,
                #         lextree=max_hyp.lextree,
                #         p_gen=max_hyp.p_gen
                #     )
                # )

                if self.use_lm:
                    lm_state, lm_scores = self.lm.predict(max_hyp.lm_state, lm_tokens)
                else:
                    lm_state = max_hyp.lm_state

                if estlm is not None:
                    estlm_scores, estlm_state = estlm(lm_tokens.unsqueeze(0), max_hyp.estlm_state)
                    estlm_scores = torch.log_softmax(estlm_scores, dim=-1)

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        # LM removed blank when being loaded
                        score += self.lm_weight * lm_scores[0][k].item()
                        if estlm is not None:   
                            score -= estlm_factor * estlm_scores[0][0][k].item()

                    # ugly hack to avoid dead repeat
                    if int(k+1) == max_hyp.yseq[-1] and int(k+1) == max_hyp.yseq[-2] and int(k+1) == max_hyp.yseq[-3]:
                        continue

                    hyps.append(
                        Hypothesis(
                            score=score,
                            vscore=0, # max_hyp.vscore + logp,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                            lextree=tree_track,
                            p_gen=max_hyp.p_gen + [store_pgen],
                            estlm_state=estlm_state,
                            trace=max_hyp.trace + [enc_pos],
                            prev_hid=new_hid
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                max_yseq = max(hyps, key=lambda x: x.score).yseq
                # kept_hyps_max = sorted([x.score for x in kept_hyps])[-2:]
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                # print(enc_pos, hyps_max, kept_hyps_max)
                if len(kept_most_prob) >= beam or (len(max_yseq) >= len(h) * 1.5 and len(kept_most_prob) > 0):
                    kept_hyps = kept_most_prob
                    break
        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, h: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        cache = {}

        if self.use_lm and not self.is_wordlm:
            B[0].lm_state = init_lm_state(self.lm_predictor)

        for hi in h:
            A = []
            C = B

            h_enc = hi.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    C,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                seq_A = [h.yseq for h in A]

                for i, hyp in enumerate(C):
                    if hyp.yseq not in seq_A:
                        A.append(
                            Hypothesis(
                                score=(hyp.score + float(beam_logp[i, 0])),
                                yseq=hyp.yseq[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        dict_pos = seq_A.index(hyp.yseq)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                        )

                if v < (self.max_sym_exp - 1):
                    if self.use_lm:
                        beam_lm_states = create_lm_batch_state(
                            [c.lm_state for c in C], self.lm_layers, self.is_wordlm
                        )

                        beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                            beam_lm_states, beam_lm_tokens, len(C)
                        )

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                yseq=(hyp.yseq + [int(k)]),
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * beam_lm_scores[i, k]

                                new_hyp.lm_state = select_lm_state(
                                    beam_lm_states, i, self.lm_layers, self.is_wordlm
                                )

                            D.append(new_hyp)

                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, h: torch.Tensor) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)

        h_length = int(h.size(0))
        u_max = min(self.u_max, (h_length - 1))

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        final = []
        cache = {}

        if self.use_lm and not self.is_wordlm:
            B[0].lm_state = init_lm_state(self.lm_predictor)

        for i in range(h_length + u_max):
            A = []

            B_ = []
            h_states = []
            for hyp in B:
                u = len(hyp.yseq) - 1
                t = i - u + 1

                if t > (h_length - 1):
                    continue

                B_.append(hyp)
                h_states.append((t, h[t]))

            if B_:
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    B_,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                h_enc = torch.stack([h[1] for h in h_states])

                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                if self.use_lm:
                    beam_lm_states = create_lm_batch_state(
                        [b.lm_state for b in B_], self.lm_layers, self.is_wordlm
                    )

                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(B_)
                    )

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if h_states[i][0] == (h_length - 1):
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.use_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]

                            new_hyp.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )

                        A.append(new_hyp)

                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = recombine_hyps(B)

        if final:
            return self.sort_nbest(final)
        else:
            return B

    def nsc_beam_search(self, h: torch.Tensor) -> List[NSCHypothesis]:
        """N-step constrained beam search implementation.

        Based and modified from https://arxiv.org/pdf/2002.03577.pdf.
        Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
        until further modifications.

        Note: the algorithm is not in his "complete" form but works almost as
        intended.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            NSCHypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens,
            beam_state,
            cache,
            self.use_lm,
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.use_lm:
            beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                None, beam_lm_tokens, 1
            )
            lm_state = select_lm_state(
                beam_lm_states, 0, self.lm_layers, self.is_wordlm
            )
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            NSCHypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=state,
                y=[beam_y[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        for hi in h:
            hyps = sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True)
            kept_hyps = []

            h_enc = hi.unsqueeze(0)

            for j, hyp_j in enumerate(hyps[:-1]):
                for hyp_i in hyps[(j + 1) :]:
                    curr_id = len(hyp_j.yseq)
                    next_id = len(hyp_i.yseq)

                    if (
                        is_prefix(hyp_j.yseq, hyp_i.yseq)
                        and (curr_id - next_id) <= self.prefix_alpha
                    ):
                        ytu = torch.log_softmax(
                            self.joint_network(hi, hyp_i.y[-1]), dim=-1
                        )

                        curr_score = hyp_i.score + float(ytu[hyp_j.yseq[next_id]])

                        for k in range(next_id, (curr_id - 1)):
                            ytu = torch.log_softmax(
                                self.joint_network(hi, hyp_j.y[k]), dim=-1
                            )

                            curr_score += float(ytu[hyp_j.yseq[k + 1]])

                        hyp_j.score = np.logaddexp(hyp_j.score, curr_score)

            S = []
            V = []
            for n in range(self.nstep):
                beam_y = torch.stack([hyp.y[-1] for hyp in hyps])

                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

                for i, hyp in enumerate(hyps):
                    S.append(
                        NSCHypothesis(
                            yseq=hyp.yseq[:],
                            score=hyp.score + float(beam_logp[i, 0:1]),
                            y=hyp.y[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )
                    )

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        score = hyp.score + float(logp)

                        if self.use_lm:
                            score += self.lm_weight * float(hyp.lm_scores[k])

                        V.append(
                            NSCHypothesis(
                                yseq=hyp.yseq[:] + [int(k)],
                                score=score,
                                y=hyp.y[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                                lm_scores=hyp.lm_scores,
                            )
                        )

                V.sort(key=lambda x: x.score, reverse=True)
                V = substract(V, hyps)[:beam]

                beam_state = self.decoder.create_batch_states(
                    beam_state,
                    [v.dec_state for v in V],
                    [v.yseq for v in V],
                )
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    V,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                if self.use_lm:
                    beam_lm_states = create_lm_batch_state(
                        [v.lm_state for v in V], self.lm_layers, self.is_wordlm
                    )
                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(V)
                    )

                if n < (self.nstep - 1):
                    for i, v in enumerate(V):
                        v.y.append(beam_y[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )
                            v.lm_scores = beam_lm_scores[i]

                    hyps = V[:]
                else:
                    beam_logp = torch.log_softmax(
                        self.joint_network(h_enc, beam_y), dim=-1
                    )

                    for i, v in enumerate(V):
                        if self.nstep != 1:
                            v.score += float(beam_logp[i, 0])

                        v.y.append(beam_y[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )
                            v.lm_scores = beam_lm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(kept_hyps)
