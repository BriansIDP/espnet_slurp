from __future__ import division

import argparse
import logging
import math
import os
from time import time
from copy import deepcopy
import random
import json

import editdistance

import numpy as np
import six
import torch

from espnet.nets.pytorch_backend.nets_utils import to_device

class SLUutils(object):
    def __init__(self, intentfile, slotfile, charlist=None, wordlevel=False, ontology=None):
        """Meeting-wise KB in decoder
        """
        self.intent2ids = {}
        self.ids2intent = []
        self.slot2ids = {'none': 0}
        self.ids2slot = ['none']
        with open(intentfile) as fin:
            for line in fin:
                intent = line.strip()
                self.intent2ids[intent] = len(self.ids2intent)
                self.ids2intent.append(intent)
        with open(slotfile) as fin:
            for line in fin:
                slot = line.strip()
                self.slot2ids[slot] = len(self.ids2slot)
                self.ids2slot.append(slot)
        self.fullwordlist = []
        if ontology is not None:
            with open(ontology) as fin:
                self.ontology = json.load(fin)
            for slottype, words in self.ontology.items():
                for word in words:
                    if word not in self.fullwordlist:
                        self.fullwordlist.append(word)
        self.nslots = len(self.ids2slot)
        self.nintents = len(self.ids2intent)
        self.char_list = charlist
        self.wordlevel = wordlevel

    def get_intent_labels(self, intents, slots, ys_pad, KBwords=[], ndistractors=1):
        intlabels = []
        slotlabels = []
        ptrlabels = []
        maxslots = 0
        slotshortlist = [[] for i in range(ys_pad.size(0))]
        slotmask = [[] for i in range(ys_pad.size(0))]
        for intent in intents:
            intlabels.append(self.intent2ids[intent])
        intlabels = torch.LongTensor(intlabels)
        for i, ys_sample in enumerate(ys_pad.tolist()):
            piece2wordmap = []
            piece2wordmap2 = []
            wordtext = ''.join(self.char_list[idx] for idx in ys_sample).replace('▁', ' ').split()
            count = 0
            for idx in ys_sample:
                if not self.wordlevel:
                    piece2wordmap.append(count)
                    piece2wordmap2.append(count)
                if idx != -1 and self.char_list[idx].endswith('▁'):
                    if self.wordlevel:
                        piece2wordmap.append(count)
                        piece2wordmap2.append(count)
                    count += 1
                elif self.wordlevel:
                    piece2wordmap.append(-1)
                    piece2wordmap2.append(count)
            wordlevel = [0] * count + [-1]
            wordlevelKB = [0] * (count + 1)
            for ent in slots[i]:
                if self.slot2ids[ent['type']] not in slotshortlist[i]:
                    slotshortlist[i].append(self.slot2ids[ent['type']])
                    mask = [0] * len(self.ids2slot)
                    mask[self.slot2ids[ent['type']]] = 1
                    slotmask[i].append(mask)
                for wordid in ent['span']:
                    wordlevel[wordid] = self.slot2ids[ent['type']]
                    if wordtext[wordid] in KBwords:
                        wordlevelKB[wordid] = 1
            slotlabel = torch.LongTensor(wordlevel)[piece2wordmap]
            slotlabels.append(slotlabel)
            if len(slotshortlist[i]) > maxslots:
                maxslots = len(slotshortlist[i])
            ptrlabels.append(torch.tensor(wordlevelKB)[piece2wordmap2])

        ndistractors = max(ndistractors, maxslots)
        for i, slotlist in enumerate(slotshortlist):
            uncovered = [slot for slot in self.ids2slot[1:] if slot not in slotlist]
            if len(slotlist) < ndistractors:
                additional = random.sample(uncovered, k=ndistractors-len(slotlist))
                for slot in additional:
                    slotshortlist[i].append(self.slot2ids[slot])
                    mask = [0] * len(self.ids2slot)
                    mask[self.slot2ids[slot]] = 1
                    slotmask[i].append(mask)

        slotmask = torch.Tensor(slotmask)
        slotlabels = torch.stack(slotlabels, dim=0)
        slotlabels = torch.cat([slotlabels, -slotlabels.new_ones(slotlabels.size(0), 1)], dim=-1)
        ptrlabels = torch.stack(ptrlabels, dim=0)
        ptrlabels = torch.cat([ptrlabels, ptrlabels.new_zeros(ptrlabels.size(0), 1)], dim=-1)
        return intlabels, slotlabels, ptrlabels, slotshortlist, slotmask

    def get_word_bound(self, yseq):
        wordbound = []
        prevpos = 1
        for i, idx in enumerate(input_ids):
            if i == 0 or i == len(input_ids) - 1:
                wordbound.append(0)
            else:
                if tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([idx])).startswith(' '):
                    prevpos = i
                wordbound.append(prevpos)
        return torch.LongTensor(wordbound)

    def predict(self, slots, intent, yseq, shortlist=[]):
        intentvalue = self.ids2intent[intent]
        # get word boundaries
        wordbound = []
        prevpos = -1
        if self.wordlevel:
            for i, idx in enumerate(yseq):
                if self.char_list[idx].endswith('▁') and (self.char_list[idx] != '▁' or (i > 0 and not self.char_list[yseq[i-1]].endswith('▁'))):
                    wordbound += [i] * (i - prevpos)
                    prevpos = i
            # wordbound.append(len(yseq)-1)
            slots = slots[wordbound]
            yseq = yseq[:len(slots)+1]

        entities = []
        entbuffer = []
        prevent = 0
        for i, ys in enumerate(yseq[:-1]):
            if slots[i] != prevent:
                if entbuffer != []:
                    ent = ''.join([self.char_list[wp] for wp in entbuffer]).strip('▁').replace('▁', ' ').lower()
                    entbuffer = []
                    entities.append({"type": self.ids2slot[prevent], "filler": ent.strip().lstrip()})
                if slots[i] != 0:
                    entbuffer.append(ys)
            elif slots[i] == prevent and slots[i] != 0:
                entbuffer.append(ys)
            prevent = slots[i]
        if entbuffer != []:
            ent = ''.join([self.char_list[wp] for wp in entbuffer]).strip('▁').replace('▁', ' ').lower()
            entbuffer = []
            entities.append({"type": self.ids2slot[prevent], "filler": ent.strip().lstrip()})
        shortlist = [self.ids2slot[slot.item()] for slot in shortlist[1] if slot.item() != 0]
        return entities, intentvalue, shortlist

    def get_slot_names(self, shortlist):
        newlist = []
        for slots in shortlist:
            wlist = []
            slots = [self.ids2slot[slot] for slot in slots if slot != 0]
            newlist.append(slots)
        return newlist

    def get_wlist_from_slots(self, shortlist, droprate=0.0):
        wlists = []
        slotwlists = []
        for slots in shortlist:
            wlist = []
            slotwlist = [[] for i in range(len(slots))]
            slots = [self.ids2slot[slot] for slot in slots if slot != 0]
            for i, slot in enumerate(slots):
                for word in self.ontology[slot]:
                    if random.random() < droprate:
                        word = random.choice(self.fullwordlist)
                    if word not in wlist:
                        wlist.append(word)
                    if word not in slotwlist[i]:
                        slotwlist[i].append(word)
            wlists.append(wlist)
            slotwlists.append(slotwlist)
        return wlists, slotwlists


class SLUNet(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, dunits, nslots, nintents, slotfactor, intentfactor, char_list, jointdim=0,
                 modmatch=False, jointrep=False, mmfactor=0.0, jointptrgen=False, mixup=0,
                 classpost=False, postfactor=0.0):
        super(SLUNet, self).__init__()

        if jointptrgen:
            dunits = dunits + 1
        if modmatch:
            self.intentproj = torch.nn.Linear(dunits, jointdim)
            # self.slotproj = torch.nn.Linear(dunits, jointdim)
            self.slotproj = torch.nn.LSTM(dunits, jointdim//2, 1, batch_first=True, bidirectional=True)
            self.slotmix = torch.nn.Linear(jointdim, jointdim)
            self.slothead = torch.nn.Linear(jointdim+(jointdim if jointrep else 0), nslots)
            self.intenthead = torch.nn.Linear(jointdim+(jointdim if jointrep else 0), nintents)
        else:
            self.slothead = torch.nn.Linear(dunits+(jointdim if jointrep else 0), nslots)
            self.intenthead = torch.nn.Linear(dunits+(jointdim if jointrep else 0), nintents)
        self.dropout = torch.nn.Dropout(0.1)
        self.nllcriterion = torch.nn.NLLLoss(ignore_index=-1, reduction='mean')
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.char_list = char_list

        self.dunits = dunits
        self.nslots = nslots
        self.nintents = nintents
        self.slotfactor = slotfactor
        self.intentfactor = intentfactor
        self.mmfactor = mmfactor
        self.modmatch = modmatch
        self.jointrep = jointrep
        self.jointdim = jointdim
        self.jointptrgen = jointptrgen
        self.mixup = mixup
        self.classpost = classpost
        self.postfactor = postfactor
        self.eos = self.char_list.index('<eos>')

    def get_word_bound(self, charlist):
        wordbound = []
        for utt in charlist:
            bound = []
            wordcount = 0
            for i, charidx in enumerate(utt):
                if i != 0 and self.char_list[utt[i-1]].endswith('▁') and self.char_list[charidx] == '▁':
                    bound.append(wordcount)
                elif charidx != -1 and self.char_list[charidx].endswith('▁'):
                    bound.append(wordcount)
                    wordcount += 1
                else:
                    bound.append(wordcount)
            bound = to_device(self, torch.tensor(bound))
            wordbound.append(bound)
        wordbound = torch.stack(wordbound)
        return wordbound

    def forward(self, d_hidden, ys_pad, intents, slots, mmemb=None, maskfactor=0.0, ptr_gen=None,
                classpost=None, factor=0.0):
        d_mask = ys_pad != -1 # (slots != -1)
        boundary_mask = (slots != -1).float()
        final_ids = d_mask.sum(dim=1)
        loss_mm = 0

        if self.jointptrgen and ptr_gen is not None:
            d_hidden = torch.cat([d_hidden, ptr_gen.unsqueeze(-1)], dim=-1)
        slot_hidden = d_hidden[:]
        lasthidden = d_hidden[torch.arange(final_ids.size(0)), final_ids]
        if mmemb is not None:
            wordbound = self.get_word_bound(torch.cat([ys_pad, ys_pad.new_ones(ys_pad.size(0), 1) * self.eos], dim=-1))
            roberta_emb = mmemb[0]
            roberta_pool = mmemb[2]
            dummy = wordbound.unsqueeze(-1).expand(wordbound.size(0), wordbound.size(1), roberta_emb.size(2))
            roberta_emb = torch.gather(roberta_emb, 1, dummy)
            if self.modmatch:
                slot_hidden = slot_hidden * torch.cat([d_mask.new_ones(d_mask.size(0), 1), d_mask], dim=-1).unsqueeze(-1)
                init_hidden = (slot_hidden.new_zeros(2, d_mask.size(0), self.jointdim//2),
                               slot_hidden.new_zeros(2, d_mask.size(0), self.jointdim//2))
                slot_hidden, memories = self.slotproj(slot_hidden, init_hidden)
                slot_hidden = self.slotmix(self.dropout(slot_hidden))
                # lasthidden = self.intentproj(lasthidden)
                lasthidden = slot_hidden[torch.arange(final_ids.size(0)), final_ids]
                # Total loss at word boundaries / batch size
                if not self.jointrep and not self.mixup > 0:
                    roberta_emb = roberta_emb.detach()
                    roberta_pool = roberta_pool.detach()
                loss_mm = (torch.abs(slot_hidden - roberta_emb).mean(dim=-1) * boundary_mask).sum() / boundary_mask.size(0)
                loss_mm += torch.abs(lasthidden - roberta_pool).mean()
                if self.mixup > 0:
                    mixmask = torch.rand(roberta_emb.size(0)) < self.mixup
                    boundary_mask = (boundary_mask * mixmask.to(boundary_mask.device).unsqueeze(-1)).unsqueeze(-1)
                    slot_hidden = boundary_mask * roberta_emb + (1 - boundary_mask) * slot_hidden
                    if random.random() < self.mixup:
                        lasthidden = roberta_pool
            if self.jointrep:
                slotmask = torch.rand(roberta_emb.size(0), roberta_emb.size(1), 1) < maskfactor
                roberta_emb = roberta_emb.masked_fill(to_device(self, slotmask), 0)
                slot_hidden = torch.cat([slot_hidden, roberta_emb], dim=-1)
                if random.random() < maskfactor:
                    roberta_pool = roberta_pool * 0
                lasthidden = torch.cat([lasthidden, roberta_pool], dim=-1)

        # For slot
        slot_logits = self.slothead(self.dropout(slot_hidden))
        if self.jointptrgen and ptr_gen is not None and self.classpost and classpost != []:
            slotdist = torch.softmax(slot_logits, dim=-1)
            ptr_gen = ptr_gen.unsqueeze(-1)
            slot_logits = ptr_gen * self.postfactor * classpost + (1 - ptr_gen * self.postfactor) * slotdist
            slot_logits = torch.log(slot_logits)
            slot_loss = self.nllcriterion(slot_logits.view(-1, slot_logits.size(-1)), slots.view(-1))
        else:
            slot_loss = self.criterion(slot_logits.view(-1, slot_logits.size(-1)), slots.view(-1))
        slot_loss *= (slots != -1).sum(dim=-1).float().mean()
        slotacc = (slot_logits.max(dim=-1)[1] == slots).sum()
        totalslots = d_mask.sum()
        slotacc = slotacc / (slots != -1).sum()

        # For intent
        intent_logits = self.intenthead(self.dropout(lasthidden))
        intent_loss = self.criterion(intent_logits, intents)
        intentacc = (intent_logits.max(dim=-1)[1] == intents).sum()
        intentacc = intentacc / d_mask.size(0)

        # calculate loss
        loss = self.slotfactor * slot_loss + self.intentfactor * intent_loss + self.mmfactor * loss_mm

        return loss, loss_mm, slotacc.item(), intentacc.item()

    def inference(self, d_hidden, y_hyp, mmemb=None, ptr_gen=None, topn=10):
        d_hidden = torch.cat(d_hidden, dim=0)
        if self.jointptrgen:
            ptr_gen = torch.clamp(to_device(self, torch.tensor(ptr_gen).unsqueeze(1)), max=1.0)
            d_hidden = torch.cat([d_hidden, ptr_gen], dim=-1)
        slot_hidden = d_hidden[:]
        lasthidden = d_hidden[-1]
        if mmemb is not None:
            if len(y_hyp) > slot_hidden.size(0):
                y_hyp = y_hyp[:slot_hidden.size(0)]
            wordbound = self.get_word_bound([y_hyp])
            roberta_emb = mmemb[0]
            roberta_pool = mmemb[2]
            roberta_emb = roberta_emb[0, wordbound[0]]
            if self.modmatch:
                init_hidden = (slot_hidden.new_zeros(2, 1, self.jointdim//2),
                               slot_hidden.new_zeros(2, 1, self.jointdim//2))
                slot_hidden, memories = self.slotproj(slot_hidden.unsqueeze(0), init_hidden)
                slot_hidden = self.slotmix(slot_hidden).squeeze(0)
                # lasthidden = torch.tanh(self.intentproj(lasthidden))
                lasthidden = slot_hidden[-1]
                # if self.mixup:
                #     slot_hidden = roberta_emb
                #     lasthidden = roberta_pool
            if self.jointrep:
                slot_hidden = torch.cat([slot_hidden, roberta_emb], dim=-1)
                lasthidden = torch.cat([lasthidden, roberta_pool[0]], dim=-1)
        # intent_logits = self.intenthead(self.intentproj(lasthidden))
        intent_logits = self.intenthead(lasthidden)
        intent_pred = intent_logits.max(dim=-1)[1].item()
        slot_logits = self.slothead(slot_hidden)
        topnslots = torch.softmax(slot_logits, dim=-1).sum(dim=0).topk(k=topn)
        # slot_logits = self.slothead(self.slotproj(slot_hidden))
        slot_pred = slot_logits.max(dim=-1)[1]
        return slot_pred, intent_pred, topnslots

    def inference_batch(self, d_hidden, ys_pad, slots, mmemb=None, topn=2, ptr_gen=None):
        d_mask = ys_pad != -1 # (slots != -1)
        boundary_mask = (slots != -1).float()
        final_ids = d_mask.sum(dim=1)
        loss_mm = 0

        with torch.no_grad():
            if self.jointptrgen and ptr_gen is not None:
                d_hidden = torch.cat([d_hidden, ptr_gen.unsqueeze(-1)], dim=-1)
            slot_hidden = d_hidden[:]
            if mmemb is not None:
                wordbound = self.get_word_bound(torch.cat([ys_pad, ys_pad.new_ones(ys_pad.size(0), 1) * self.eos], dim=-1))
                roberta_emb = mmemb[0]
                dummy = wordbound.unsqueeze(-1).expand(wordbound.size(0), wordbound.size(1), roberta_emb.size(2))
                roberta_emb = torch.gather(roberta_emb, 1, dummy)
                if self.modmatch:
                    slot_hidden = slot_hidden * torch.cat([d_mask.new_ones(d_mask.size(0), 1), d_mask], dim=-1).unsqueeze(-1)
                    init_hidden = (slot_hidden.new_zeros(2, d_mask.size(0), self.jointdim//2),
                                   slot_hidden.new_zeros(2, d_mask.size(0), self.jointdim//2))
                    slot_hidden, memories = self.slotproj(slot_hidden, init_hidden)
                    slot_hidden = self.slotmix(self.dropout(slot_hidden))
                if self.jointrep:
                    slot_hidden = torch.cat([slot_hidden, roberta_emb], dim=-1)
            slot_logits = self.slothead(self.dropout(slot_hidden))
            topnslots = torch.softmax(slot_logits, dim=-1).sum(dim=1).topk(k=topn)
        topnslots = topnslots[1].tolist()
        return topnslots

class SLUGenutils(object):
    def __init__(self, slotfile, connection, eos, charlist=None, ndistractors=1, ontofile=None,
                 simpletod=False):
        """Meeting-wise KB in decoder
        """
        self.slot2bpe = {}
        self.slot2ids = {}
        self.char_list = charlist
        self.simpletod = simpletod
        self.joiner = ','
        # if simpletod:
        #     self.char_list.append(',▁')
        #     self.joiner = len(self.char_list) - 1
        self.char_dict = {c: i for i, c in enumerate(charlist)}
        self.eos = eos
        self.slotorder = []
        with open(slotfile) as fin:
            for line in fin:
                slotelems = line.split()
                slotbpe = [self.char_list.index(wp) for wp in slotelems[1:]]
                self.slot2ids[slotelems[0]] = len(self.slotorder)
                self.slotorder.append(slotelems[0])
                self.slot2bpe[slotelems[0]] = slotbpe
        self.nslots = len(self.slot2bpe.keys())
        with open(connection) as fin:
            lines = fin.readlines()
            self.delimiter = [self.char_list.index(wp) for wp in lines[0].split()]
            self.connector = [self.char_list.index(wp) for wp in lines[1].split()]
            self.none = [self.char_list.index(wp) for wp in lines[2].split()]
            self.connector_str = ''.join(lines[1].split()).replace('▁', '').lower()
            self.delimiter_str = ''.join(lines[0].split()).replace('▁', '').lower()
            self.none_str = ''.join(lines[2].split()).replace('▁', ' ').strip().lower()
        self.distractors = {}
        self.distractors_str = {}
        for slottype, slotbpe in self.slot2bpe.items():
            self.distractors[slottype] = ([self.eos] + slotbpe + self.delimiter + self.none + [self.eos],
                [0] * (len(slotbpe) + len(self.delimiter)+1) + [1] * (len(self.none)+1))
            self.distractors_str[slottype] = ' '.join(slottype.split('_') + ['is', self.none_str])

        self.ontology = None
        self.fullwordlist = []
        if ontofile is not None and ontofile != '':
            with open(ontofile) as fin:
                self.ontology = json.load(fin)
            for slottype, words in self.ontology.items():
                for word in words:
                    if word not in self.fullwordlist:
                        self.fullwordlist.append(word)

        self.ndistractors = ndistractors
        self.get_slot_queries()

    def get_labeldict(self, ys_sample, slots):
        labeldict = {}
        word2piecemap = {0:[]}
        count = 0
        for idx in ys_sample:
            if idx != -1 and self.char_list[idx].endswith('▁'):
                word2piecemap[count].append(idx)
                count += 1
                word2piecemap[count] = []
            elif idx != -1:
                word2piecemap[count].append(idx)
        for ent in slots:
            # curr_label = self.slot2bpe[ent['type']]
            if ent['type'] in labeldict:
                labeldict[ent['type']].extend(self.connector)
            else:
                labeldict[ent['type']] = []
            for wordid in ent['span']:
                labeldict[ent['type']].extend(word2piecemap[wordid])
        return labeldict

    def get_labeldict_direct(self, slots):
        labeldict = {}
        for ent in slots:
            if ent['type'] in labeldict:
                labeldict[ent['type']].extend(self.connector)
            else:
                labeldict[ent['type']] = []
            for charstr in ent['value'].split():
                labeldict[ent['type']].append(self.char_dict[charstr])
        return labeldict

    def get_simpletod_labels(self, slots, ys_pad):
        slotlabels = []
        slotmasks = []
        slotsamplemap = []
        slottext = []
        slotlist = []
        classprobmask = []
        maxlen = 0
        for i, slotentities in enumerate(slots):
            labeldict = self.get_labeldict_direct(slotentities)
            slotlist.append([])
            slotlabels.append([self.eos])
            for enttype, entvalues in labeldict.items():
                slotlabels[-1].extend(self.slot2bpe[enttype] + self.delimiter + entvalues)
                slotlabels[-1].append(self.joiner)
                if entvalues != self.none:
                    slotlist[-1].append(enttype)
            slotstr = ''.join([self.char_list[idx] for idx in slotlabels[-1][1:-1]]).replace('▁', ' ').strip().lower()
            slottext.append(slotstr)
            slotlabels[-1] = slotlabels[-1][:-1]
            slotlabels[-1].append(self.eos)
            slotmasks.append([1] * len(slotlabels[-1]))
            if maxlen < len(slotlabels[-1]):
                maxlen = len(slotlabels[-1])
            slotsamplemap.append(i)
        # Pad slot seqs
        for k, slotlabel in enumerate(slotlabels):
            slotlabels[k] += [0] * (maxlen - len(slotlabels[k]))
            slotmasks[k] += [0] * (maxlen - len(slotmasks[k]))
        slotlabels = torch.LongTensor(slotlabels)
        slotmasks = torch.LongTensor(slotmasks)
        slotsamplemap = torch.LongTensor(slotsamplemap)
        return slotlabels, slotmasks, slotsamplemap, slottext, slotlist, classprobmask

    def get_slot_labels(self, slots, ys_pad, fullslottext=False, topk=1):
        slotlabels = []
        slotmasks = []
        slotsamplemap = []
        slottext = []
        slotlist = []
        classprobmask = [[] for i in range(ys_pad.size(0))]
        maxlen = 0
        finals = (ys_pad != -1).sum(dim=1)
        for i, ys_sample in enumerate(ys_pad):
            ys_sample = ys_sample.tolist() if not isinstance(ys_sample, list) else ys_sample
            if len(slots[i]) > 0 and 'span' in slots[i][0]:
                labeldict = self.get_labeldict(ys_sample, slots[i])
            else:
                labeldict = self.get_labeldict_direct(slots[i])
            slotlist.append([])
            slotcount = 0
            if not self.simpletod or random.random() < 0.5:
                for enttype, entvalues in labeldict.items():
                    slotlabels.append([self.eos] + self.slot2bpe[enttype] + self.delimiter + entvalues + [self.eos])
                    slotmasks.append([0] * (len(self.slot2bpe[enttype])+len(self.delimiter)+1) + [1] * (len(entvalues)+1))
                    if fullslottext:
                        slottext.append(''.join([self.char_list[idx] for idx in slotlabels[-1][1:-1]]).replace('▁', ' ').strip().lower())
                    else:
                        slottext.append(self.slotquerydict[enttype])
                    if maxlen < len(slotlabels[-1]):
                        maxlen = len(slotlabels[-1])
                    slotsamplemap.append(i)
                    slotcount += 1
                    slotlist[-1].append(enttype)
                    mask = [0] * len(self.slotorder)
                    mask[self.slot2ids[enttype]] = 1
                    classprobmask[i].append(mask)
            # add distractors
            uncovered = [label for label in self.distractors.keys() if label not in labeldict]
            if slotcount < self.ndistractors:
                randomslots = random.sample(uncovered, min(len(uncovered), self.ndistractors-slotcount))
                for randomslot in randomslots:
                    slotlabels.append(self.distractors[randomslot][0])
                    slotmasks.append(self.distractors[randomslot][1])
                    if fullslottext:
                        slottext.append(self.distractors_str[randomslot])
                    else:
                        slottext.append(self.slotquerydict[randomslot])
                    if maxlen < len(slotlabels[-1]):
                        maxlen = len(slotlabels[-1])
                    slotsamplemap.append(i)
                    # slotlist[-1].append(randomslot)
                    mask = [0] * len(self.slotorder)
                    mask[self.slot2ids[randomslot]] = 1
                    classprobmask[i].append(mask)
            if len(slotlist[-1]) < topk:
                slotlist[-1] = slotlist[-1] + randomslots[:(topk - len(slotlist[-1]))]
            elif len(slotlist[-1]) > topk:
                slotlist[-1] = slotlist[-1][:topk]
        # Pad slot seqs
        for k, slotlabel in enumerate(slotlabels):
            slotlabels[k] += [0] * (maxlen - len(slotlabels[k]))
            slotmasks[k] += [0] * (maxlen - len(slotmasks[k]))
        slotlabels = torch.LongTensor(slotlabels)
        slotmasks = torch.LongTensor(slotmasks)
        return slotlabels, slotmasks, torch.LongTensor(slotsamplemap), slottext, slotlist, classprobmask

    def get_slot_queries(self):
        self.slotqueries = []
        self.slotquerytext = []
        self.slotquerydict = {}
        for slotname in self.slotorder:
            slotbpe = self.slot2bpe[slotname]
            self.slotqueries.append(torch.LongTensor([self.eos] + slotbpe + self.delimiter))
            querytext = ''.join([self.char_list[bpe] for bpe in self.slotqueries[-1][1:]]).replace('▁', ' ').strip().lower()
            self.slotquerytext.append(querytext)
            self.slotquerydict[slotname] = querytext

    def predict(self, slots):
        slotpairs = []
        for slotidx, slotname in enumerate(self.slotorder):
            slotspred = ''.join([self.char_list[idx] for idx in slots[slotidx]]).replace('▁', ' ').strip().lower()
            if not slotspred.startswith(self.none_str) and slotspred != 'n':
                slotspred = slotspred.split(' '+self.connector_str+' ')
                for slot in slotspred:
                    slotpairs.append({'type': slotname, 'filler': slot})
        return slotpairs

    def predict_simpletod(self, slots):
        slottext = ''.join([self.char_list[idx] for idx in slots[:-1]]).replace('▁', ' ').strip().lower()
        slotpairs = []
        for pairs in slottext.split(' , '):
            slottype, slotspred = pairs.split(' '+self.delimiter_str+' ', 1)
            slottype = '_'.join(slottype.split())
            if not slotspred.startswith(self.none_str) and slottype in self.slotorder:
                slotpairs.append({'type': slottype, 'filler': slotspred})
        return slotpairs

    def get_wlist_from_slots(self, shortlist, droprate=0.0, topn=1):
        wlists = []
        slotwlists = []
        for slots in shortlist:
            # slots = slots[:topn]
            wlist = []
            slotwlist = [[] for i in range(len(slots))]
            # slots = [self.ids2slot[slot] for slot in slots if slot != 0]
            for i, slot in enumerate(slots):
                for word in self.ontology[slot]:
                    if random.random() < droprate:
                        word = random.choice(self.fullwordlist)
                    if word not in wlist:
                        wlist.append(word)
                    if word not in slotwlist[i]:
                        slotwlist[i].append(word)
            wlists.append(wlist)
            slotwlists.append(slotwlist)
        return wlists, slotwlists

    def get_copy_labels(self, wlists, slottext, slotlabels, slotmap, slotmask, entity=False):
        labels = []
        appeared_words = [[] for i in range(len(wlists))]
        appeared_words_per_slot = []
        for i, slotlabel in enumerate(slotlabels):
            if not slottext[i].endswith(self.none_str):
                wordbuffer = []
                wordbuffers = []
                words_per_slot = []
                label = []
                for k, charid in enumerate(slotlabel):
                    if slotmask[i, k] == 1:
                        wordbuffer.append(self.char_list[charid])
                    else:
                        label.append(0)
                    if wordbuffer != [] and (self.char_list[charid].endswith('▁') or self.char_list[charid] == '<eos>'):
                        wordbuffers.append(wordbuffer)
                        if entity:
                            label.extend([0] * len(wordbuffer))
                            count = len(wordbuffers) - 1
                            entitybuffer = []
                            while count >= 0:
                                entitybuffer = wordbuffers[count] + entitybuffer
                                if tuple(entitybuffer) in wlists[slotmap[i]]:
                                    label[-len(entitybuffer):] = [1] * len(entitybuffer)
                                    appeared_words[slotmap[i]].append(tuple(entitybuffer))
                                    words_per_slot.append(tuple(entitybuffer))
                                count -= 1
                        elif tuple(wordbuffer) in wlists[slotmap[i]]:
                            label.extend([1] * len(wordbuffer))
                            appeared_words[slotmap[i]].append(tuple(wordbuffer))
                            words_per_slot.append(tuple(wordbuffer))
                        else:
                            label.extend([0] * len(wordbuffer))
                        wordbuffer = []
                labels.append(torch.tensor(label).to(slotlabel.device))
                appeared_words_per_slot.append(words_per_slot)
            else:
                labels.append(slotlabel.new_zeros(slotlabel.size()))
                appeared_words_per_slot.append([])
        labels = torch.stack(labels, dim=0)
        return labels, appeared_words, appeared_words_per_slot


class SLUGenNet(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, dunits, nslots, slotfactor, embdim, char_list, preLMdim=0,
                 modmatch=False, jointrep=False, tcpgen=False, attndim=1, embsize=1, copylossfac=0.0,
                 use_gpt_gen=False, connector=None, jointptrgen=False, nonestr=None, history=False,
                 memnet=False):
        super(SLUGenNet, self).__init__()

        if jointrep:
            self.jointdim = dunits + preLMdim
        else:
            self.jointdim = dunits
        if jointptrgen:
            self.generator = torch.nn.LSTM(self.jointdim+1, self.jointdim, 1, batch_first=True)
        else:
            self.generator = torch.nn.LSTM(self.jointdim, self.jointdim, 1, batch_first=True)
        self.tcpgen = tcpgen
        self.jointptrgen = jointptrgen
        self.use_gpt_gen = use_gpt_gen
        self.connector = connector
        self.nonestr = nonestr
        self.memnet = memnet
        if self.tcpgen:
            self.embed = torch.nn.Embedding(len(char_list), embdim)
            if self.memnet:
                self.charlstm = torch.nn.LSTM(embdim, attndim, 1, batch_first=True)
                self.memnetlayer = torch.nn.Linear(self.jointdim + attndim, self.jointdim)
                self.Qproj = torch.nn.Linear(self.jointdim, attndim)
            else:
                self.gcn_l1 = torch.nn.Linear(embdim, embsize)
                self.gcn_l2 = torch.nn.Linear(embsize, embsize)
                self.ooKBemb = torch.nn.Embedding(1, embsize)
                self.pointer_gate = torch.nn.Linear(attndim + self.jointdim, 1)
                self.Qproj = torch.nn.Linear(self.jointdim, attndim)
                self.Kproj = torch.nn.Linear(embsize, attndim)

        self.copylossfac = copylossfac
        if self.copylossfac > 0:
            self.copynet = torch.nn.Linear(self.jointdim, 2)

        self.dropout = torch.nn.Dropout(0.1)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.nllcriterion = torch.nn.NLLLoss(ignore_index=-1, reduction='mean')
        self.decoder = torch.nn.Linear(self.jointdim, len(char_list))
        self.char_list = char_list
        self.history = history

        self.dunits = dunits
        self.embdim = embdim
        self.residual = self.dunits != self.embdim
        self.nslots = nslots
        self.slotfactor = slotfactor
        self.modmatch = modmatch
        self.jointrep = jointrep
        self.attndim = attndim
        self.embsize = embsize
        self.eos = self.char_list.index('<eos>')

    def init_hidden(self, bsz):
        return (self.decoder.weight.new_zeros(1, bsz, self.jointdim),
                self.decoder.weight.new_zeros(1, bsz, self.jointdim))

    def init_hidden_memnet(self, bsz):
        return (self.decoder.weight.new_zeros(1, bsz, self.attndim),
                self.decoder.weight.new_zeros(1, bsz, self.attndim))

    def get_lextree_encs_gcn(self, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[0] == {} and wordpiece is not None:
            idx = len(embeddings)
            # ey = self.embed(to_device(self, torch.LongTensor([wordpiece])))
            ey = self.embed.weight[wordpiece].unsqueeze(0)
            embeddings.append(self.dropout(ey))
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
                embeddings.append(self.dropout(ey))
            for newpiece, values in lextree[0].items():
                ids.append(self.get_lextree_encs_gcn(values, embeddings, adjacency, newpiece))
            if wordpiece is not None:
                adjacency.append([idx] + ids)
            lextree.append(ids)
            lextree.append(idx)
            return idx

    def forward_gcn(self, lextree, embeddings, adjacency):
        n_nodes = len(embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        adjacency_mat = embeddings.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0
        degrees = torch.diag(torch.sum(adjacency_mat, dim=-1) ** -0.5)
        nodes_encs = self.gcn_l1(embeddings)
        adjacency_mat = torch.einsum('ij,jk->ik', degrees, adjacency_mat)
        adjacency_mat = torch.einsum('ij,jk->ik', adjacency_mat, degrees)
        nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, nodes_encs))
        nodes_encs = self.gcn_l2(nodes_encs)
        nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, nodes_encs))
        return nodes_encs

    def fill_lextree_encs_gcn(self, lextree, nodes_encs, wordpiece=None):
        if lextree[0] == {} and wordpiece is not None:
            idx = lextree[4]
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs_gcn(values, nodes_encs, newpiece)
            lextree[3] = nodes_encs[idx].unsqueeze(0)

    def encode_tree(self, prefixtree):
        embeddings, adjacency = [], []
        self.get_lextree_encs_gcn(prefixtree, embeddings, adjacency)
        nodes_encs = self.forward_gcn(prefixtree, embeddings, adjacency)
        self.fill_lextree_encs_gcn(prefixtree, nodes_encs)

    def remap_query(self, gptemb, charlist, pad_id=-1):
        newhidden = []
        for i, utt in enumerate(charlist):
            if self.history: # The first vector encodes all history
                wordcount = 1
                newhid = [gptemb[i][0:1]]
            else:
                wordcount = 0
                newhid = [gptemb[i].new_zeros(1, gptemb[i].size(-1))]
            for j, charidx in enumerate(utt):
                if charidx != pad_id and self.char_list[charidx].endswith('▁'):
                    if wordcount >= gptemb[i].size(0):
                        newhid.append(gptemb[i].new_zeros(1, gptemb[i].size(-1)))
                    else:
                        newhid.append(gptemb[i][wordcount:wordcount+1])
                    wordcount += 1
                else:
                    newhid.append(gptemb[i].new_zeros(1, gptemb[i].size(-1)))
            newhidden.append(torch.cat(newhid))
        newhidden = torch.stack(newhidden)
        return newhidden

    def get_step_embs_inference(self, char_idx, tree_track, reset_tree):
        ooKB_id = len(self.char_list)
        new_tree = tree_track[0]
        char_idx = char_idx if isinstance(char_idx, int) else char_idx.item()
        ptr_gen = True
        if char_idx in [self.eos]:
            new_tree = reset_tree # self.meetingKB.meetinglextree[meeting] if isinstance(meeting, str) else meeting
        elif self.char_list[char_idx].endswith('▁'):
            if char_idx in new_tree and new_tree[char_idx][0] != {}:
                new_tree = new_tree[char_idx]
            else:
                new_tree = reset_tree
        elif char_idx in new_tree:
            new_tree = new_tree[char_idx]
        else:
            new_tree = [{}]
            ptr_gen = False
        indices = list(new_tree[0].keys())
        if len(new_tree) > 2 and new_tree[0] != {}:
            step_embs = torch.cat([new_tree[0][key][3] for key in indices], dim=0)
        else:
            step_embs = torch.empty(0, self.embsize)
        back_transform = []
        indices += [ooKB_id] # list(new_tree[0].keys()) + [ooKB_id]
        for i, ind in enumerate(indices):
            one_hot = [0] * (ooKB_id + 1)
            one_hot[ind] = 1
            back_transform.append(one_hot)
        back_transform = torch.Tensor(back_transform)
        # step_embs = torch.einsum('jk,km->jm', back_transform, meeting_KB)
        step_embs = torch.cat([step_embs, self.ooKBemb.weight], dim=0).unsqueeze(0)
        step_mask = torch.zeros(back_transform.size(0)).byte().unsqueeze(0)
        step_embs = step_embs.unsqueeze(0)
        step_mask = step_mask.unsqueeze(0)
        back_transform = back_transform.unsqueeze(0).unsqueeze(0)
        return step_mask, new_tree, ptr_gen, step_embs, back_transform

    def forward_tcpgen_inference(self, char_idx, tree_track, reset_tree, query):
        step_mask, tree_track, inKB, step_embs, back_transform = self.get_step_embs_inference(
            char_idx, tree_track, reset_tree)
        query = query.unsqueeze(0).unsqueeze(0)
        ptr_dist, KBembedding = self.get_slotKB_emb_map(query, step_embs, step_mask, back_transform)
        if not inKB:
            p_gen = torch.zeros(1)
        else:
            p_gen = torch.sigmoid(self.pointer_gate(torch.cat([KBembedding, query], dim=-1)))
            p_gen = p_gen.squeeze(0).squeeze(0)
        ptr_dist = ptr_dist.squeeze(0).squeeze(0)
        return ptr_dist, p_gen, tree_track, KBembedding

    def get_step_embs(self, trees, slots, slotmap):
        ooKB_id = len(self.char_list)
        maxlen = 0
        index_list = []
        p_gen_masks = []
        step_embs = []
        for n, char_ids in enumerate(slots):
            tree = trees[slotmap[n]]
            indices = []
            stepemb = []
            p_gen_mask = []
            char_ids = char_ids.tolist()
            for i, vy in enumerate(char_ids):
                tree = tree[0]
                if vy == self.eos:
                    tree = trees[slotmap[n]]
                    p_gen_mask.append(0)
                elif vy > 0 and self.char_list[vy].endswith('▁'):
                    if vy in tree and tree[vy][0] != {}:
                        tree = tree[vy]
                    else:
                        tree = trees[slotmap[n]]
                    p_gen_mask.append(0)
                elif vy in tree:
                    tree = tree[vy]
                    p_gen_mask.append(0)
                else:
                    tree = [{}]
                    p_gen_mask.append(1)
                if len(tree[0].keys()) > maxlen:
                    maxlen = len(tree[0].keys())
                indices.append(list(tree[0].keys()))
                # if len(tree) > 3 and tree[3] != []:
                if len(tree) > 2 and tree[0] != {}:
                    stepemb.append(torch.cat([tree[0][key][3] for key in indices[-1]]))
                    # stepemb.append(nodes_encs[n][tree[3]])
                else:
                    stepemb.append(to_device(self, torch.empty(0, self.embsize)))
            step_embs.append(stepemb)
            p_gen_masks.append(p_gen_mask)
            index_list.append(indices)
        maxlen += 1
        step_mask = []
        back_transform = to_device(self, torch.zeros(slots.size(0), slots.size(1), maxlen, ooKB_id+1))
        ones_mat = to_device(self, torch.ones(back_transform.size()))
        for i, indices in enumerate(index_list):
            mask = []
            for j, index in enumerate(indices):
                mask.append(len(index) * [0] + (maxlen - len(index) - 1) * [1] + [0])
                pad_embs = self.ooKBemb.weight.repeat(maxlen-len(index), 1)
                index += [ooKB_id] * (maxlen-len(index))
                step_embs[i][j] = torch.cat([step_embs[i][j], pad_embs], dim=0)
            step_mask.append(mask)
            step_embs[i] = torch.stack(step_embs[i])
        step_mask = to_device(self, torch.tensor(step_mask).byte())
        index_list = to_device(self, torch.LongTensor(index_list))
        back_transform.scatter_(dim=-1, index=index_list.unsqueeze(-1), src=ones_mat)
        step_embs = torch.stack(step_embs)
        p_gen_masks = torch.tensor(p_gen_masks).to(step_embs.device)
        return step_embs, step_mask, back_transform, p_gen_masks

    def get_slotKB_emb_map(self, query, slot_embs, slot_mask, back_transform):
        query = self.dropout(self.Qproj(query))
        slot_KB = self.dropout(self.Kproj(slot_embs))
        KBweight = torch.einsum('bijk,bik->bij', slot_KB, query)
        KBweight = KBweight / math.sqrt(query.size(-1))
        # if self.memnet:
        # slot_mask[:, :, -1] = 1
        KBweight.masked_fill_(slot_mask.bool(), -1e9)
        KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
        if slot_KB.size(2) > 1:
            KBembedding = torch.einsum('bijk,bij->bik', slot_KB[:,:,:-1,:], KBweight[:,:,:-1])
        else:
            KBembedding = KBweight.new_zeros(slot_KB.size(0), slot_KB.size(1), slot_KB.size(-1))
        KBweight = torch.einsum('bijk,bij->bik', back_transform, KBweight)
        return KBweight, KBembedding

    def get_mask(self, gptemb, maskfactor, slotmap):
        zero_mask = gptemb.new_zeros(gptemb.size(0), 1, 1)
        one_mask = torch.rand(gptemb.size(0), 1, 1) < maskfactor
        expand_mask = one_mask[slotmap]
        one_mask = one_mask.repeat(1, gptemb.size(1)-1, 1)
        one_mask = to_device(self, one_mask)
        combined_mask = torch.cat([zero_mask, one_mask], dim=1)
        return combined_mask, expand_mask

    def memnet_encode(self, wordlists, slotoutput, slotmap):
        wordidlist = []
        maxlen = 0
        finalids = []
        attnmasks = []
        wordspersample = max([len(wlist) for wlist in wordlists])
        for wlist in wordlists:
            finalids.append([])
            wordidlist.append([])
            for word in wlist:
                charlist = [self.char_list.index(wp) if wp in self.char_list else 0 for wp in word]
                finalids[-1].append(len(charlist) - 1)
                if len(charlist) > maxlen:
                    maxlen = len(charlist)
                wordidlist[-1].append(charlist)
            attnmasks.append([0] * len(wordidlist[-1]) + [1] * (wordspersample-len(wordidlist[-1])))
            if len(wordidlist[-1]) < wordspersample:
                finalids[-1] += [0 for i in range(wordspersample-len(wordidlist[-1]))]
                wordidlist[-1] += [[0] for i in range(wordspersample-len(wordidlist[-1]))]
        for widlist in wordidlist:
            for word in widlist:
                if len(word) < maxlen:
                    word += [0] * (maxlen - len(word))
        finalids = torch.LongTensor(finalids).to(slotoutput.device)
        wordidlist = torch.LongTensor(wordidlist).to(slotoutput.device)
        attnmasks = torch.tensor(attnmasks).to(slotoutput.device)
        # forward word encodings
        wordencoding = self.embed(wordidlist)
        init_hidden = self.init_hidden_memnet(wordidlist.size(0)*wordidlist.size(1))
        wordencoding, _ = self.charlstm(wordencoding.view(-1, wordencoding.size(2), wordencoding.size(3)), init_hidden)
        wordencoding = wordencoding[torch.arange(wordencoding.size(0)), finalids.view(-1)]
        wordencoding = wordencoding.view(wordidlist.size(0), wordidlist.size(1), -1)
        # bsize * nwords * embdim
        wordencoding = wordencoding[slotmap]
        attnmasks = attnmasks[slotmap].bool().unsqueeze(1)
        query = self.Qproj(slotoutput)
        # nquery * seqlen * attndim, nquery * nwords * attndim -> nquery * seqlen * nwords
        attn_weight = torch.einsum('ijk,ink->ijn', query, wordencoding)
        attn_weight = torch.softmax(attn_weight.masked_fill(attnmasks, -1e9), dim=-1)
        # nquery * seqlen * nwords, nquery * nwords * attndim -> nquery * seqlen * attndim
        output_emb = torch.einsum('ijn,ink->ijk', attn_weight, wordencoding)
        return output_emb

    def forward(self, d_hidden, ys_pad, slots, slotmask, slotmap, queryemb, gpthidden=None, trees=None,
                node_encs=None, slothidden=None, ptr_gen=None, maskfactor=0.0, copylabel=None):
        d_mask = ys_pad != -1 # (slots != -1)
        final_ids = d_mask.sum(dim=1)

        if self.jointrep:
            gptemb = self.remap_query(gpthidden, ys_pad)
            gptmask, gptmask_expand = self.get_mask(gptemb, maskfactor, slotmap)
            gptemb = gptemb.masked_fill(gptmask.bool(), 0)
            d_hidden = self.dropout(torch.cat([d_hidden, gptemb], dim=-1))
            if self.use_gpt_gen:
                slotgptemb = self.remap_query(slothidden, slots[:,1:], pad_id=0)
                gptmask = torch.rand(slotgptemb.size(0), 1, 1) < maskfactor
                slotgptemb = slotgptemb.masked_fill(to_device(self, gptmask), 0)
                if self.residual:
                    residual_pad = queryemb.new_zeros(queryemb.size(0), queryemb.size(1), self.dunits-self.embdim)
                    queryemb = torch.cat([queryemb, residual_pad, slotgptemb], dim=-1)
                else:
                    queryemb = torch.cat([queryemb, slotgptemb], dim=-1)
        else:
            d_hidden = self.dropout(d_hidden)
            if self.residual:
                residual_pad = queryemb.new_zeros(queryemb.size(0), queryemb.size(1), self.dunits-self.embdim)
                queryemb = torch.cat([queryemb, residual_pad], dim=-1)

        d_hidden = d_hidden[slotmap]
        final_ids = final_ids[slotmap]
        if self.jointptrgen and ptr_gen is not None:
            ptr_gen = ptr_gen.detach()
            d_hidden = torch.cat([d_hidden, ptr_gen.unsqueeze(-1)], dim=-1)
            queryemb = torch.cat([queryemb, ptr_gen.new_zeros(queryemb.size(0), queryemb.size(1), 1)], dim=-1)

        carry_hidden = []
        init_hidden = self.init_hidden(d_hidden.size(0))
        for pos in range(d_hidden.size(1)):
            slot_output, init_hidden = self.generator(d_hidden[:,pos:pos+1], init_hidden)
            carry_hidden.append(init_hidden)
        carry_hidden = list(zip(*carry_hidden))
        carry_hidden[0] = torch.cat(carry_hidden[0], dim=0).transpose(0, 1)[torch.arange(final_ids.size(0)), final_ids].unsqueeze(0)
        carry_hidden[1] = torch.cat(carry_hidden[1], dim=0).transpose(0, 1)[torch.arange(final_ids.size(0)), final_ids].unsqueeze(0)
        # carry_hidden[0] = torch.index_select(carry_hidden[0][torch.arange(final_ids.size(0)), final_ids], 0, slotmap).unsqueeze(0)
        # carry_hidden[1] = torch.index_select(carry_hidden[1][torch.arange(final_ids.size(0)), final_ids], 0, slotmap).unsqueeze(0)

        queryemb = self.dropout(queryemb)
        slot_output, _ = self.generator(queryemb, carry_hidden)

        if self.tcpgen and trees is not None:
            if self.memnet:
                KBembedding = self.memnet_encode(trees, slot_output, slotmap)
                slot_output = torch.tanh(self.memnetlayer(torch.cat([slot_output, KBembedding], dim=-1)))
            else:
                stepembs, stepmasks, backtransform, p_gen_mask = self.get_step_embs(trees, slots, slotmap)
                ptr_dist, KBembedding = self.get_slotKB_emb_map(slot_output, stepembs, stepmasks, backtransform)
                p_gen = torch.sigmoid(self.pointer_gate(torch.cat([KBembedding, slot_output], dim=-1)))
                ptr_dist = ptr_dist[:, :-1].contiguous()
                p_gen = p_gen.masked_fill(p_gen_mask.bool().unsqueeze(-1), 0)[:,:-1].contiguous()

        copyloss = torch.tensor(0)
        if self.copylossfac > 0 and copylabel is not None:
            if self.tcpgen and trees is not None and not self.memnet:
                p_gen_sum = ptr_dist[:,:,:-1].sum(dim=-1) * p_gen.squeeze(-1)
                copyloss = (-torch.log(p_gen_sum + 1e-9) * copylabel[:,1:]).sum() / max(1.0, copylabel.sum())
            elif not self.tcpgen:
                copylogits = self.copynet(slot_output)[:, :-1].contiguous()
                copylabel = copylabel.masked_fill((1 - slotmask).bool(), -1)[:, 1:].contiguous()
                copyloss = self.criterion(copylogits.view(-1, copylogits.size(2)), copylabel.view(-1))

        slot_output = self.decoder(slot_output)[:, :-1].contiguous()

        # Calculate loss
        slots = slots.masked_fill((1 - slotmask).bool(), -1)[:, 1:].contiguous()
        if self.tcpgen and trees is not None and not self.memnet:
            slot_output = torch.softmax(slot_output, dim=-1)
            ptr_gen_complement = ptr_dist[:,:,-1:] * p_gen
            slot_output = (1 - p_gen + ptr_gen_complement) * slot_output + ptr_dist[:,:,:-1] * p_gen
            slot_output = torch.log(slot_output)
            slot_loss = self.nllcriterion(slot_output.view(-1, slot_output.size(2)), slots.view(-1))
        else:
            slot_loss = self.criterion(slot_output.view(-1, slot_output.size(2)), slots.view(-1))
        slot_loss *= (slots != -1).sum(dim=-1).float().mean()
        totalslots = slotmask.sum()
        loss = self.slotfactor * slot_loss
        if self.copylossfac > 0 and copylabel is not None:
            loss += copyloss * self.copylossfac

        # calculate acc
        slotacc = (slot_output.max(dim=-1)[1] == slots) * slotmask[:, 1:]
        slotacc = slotacc.sum() / slotmask.sum()
        return loss, slotacc.item(), copyloss.item()

    def get_wordtext(self, charlist):
        text = []
        for charidx in charlist:
            text.append(self.char_list[charidx])
        text = ''.join(text).replace('▁', ' ').strip().lower()
        return text

    def inference(self, d_hidden, decoder_emb, queryemb=None, gpthidden=None, yseq=[],
                  slothidden=None, slotids=None, ptr_gen=None, gptmodel=None, pasttext=(),
                  trees=None):
        maxlen = max(3, 2 * len(d_hidden))
        rectext, querytext = pasttext
        if isinstance(d_hidden, list):
            d_hidden = torch.cat(d_hidden, dim=0)
        if self.jointrep:
            gpthidden = self.remap_query(gpthidden, [yseq])
            if gpthidden.size(1) > d_hidden.size(0):
                gpthidden = gpthidden[:, :d_hidden.size(0)]
            d_hidden = torch.cat([d_hidden, gpthidden.squeeze(0)], dim=-1)

        init_hidden = self.init_hidden(1)
        if self.jointptrgen and ptr_gen is not None and len(ptr_gen) > 0:
            if isinstance(ptr_gen[0], float):
                ptr_gen = torch.tensor(ptr_gen).unsqueeze(1).repeat(1, len(queryemb))
            else:
                ptr_gen = torch.stack(ptr_gen)[:,1:]
        else:
            slot_output, reset_hidden = self.generator(d_hidden.unsqueeze(0), init_hidden)
        slot_output = []

        for i, query in enumerate(queryemb):
            if self.jointptrgen and ptr_gen is not None:
                _, reset_hidden = self.generator(torch.cat([d_hidden, ptr_gen[:,i:i+1]], dim=-1).unsqueeze(0), init_hidden)
            if self.use_gpt_gen:
                slotgptemb = self.remap_query(slothidden[i:i+1], [slotids[i][1:]], pad_id=0)
                # query = self.slotjointproj(torch.cat([query, slotgptemb.squeeze(0)], dim=-1))
                if self.residual:
                    query_pad = query.new_zeros(query.size(0), self.dunits-self.embdim)
                    # ilm
                    # query_ilm = torch.cat([query, query_pad, slotgptemb.squeeze(0)], dim=-1)
                    query = torch.cat([query, query_pad, slotgptemb.squeeze(0)], dim=-1)
                else:
                    query = torch.cat([query, slotgptemb.squeeze(0)], dim=-1)
            if self.jointptrgen and ptr_gen is not None:
                query = torch.cat([query, query.new_zeros(query.size(0), 1)], dim=-1)
                # ilm
                # query_ilm = torch.cat([query_ilm, query.new_zeros(query.size(0), 1)], dim=-1)
            next_output, query_init_hidden = self.generator(query.unsqueeze(0), reset_hidden)
            # ilm
            # ilm_output, query_ilm_hidden = self.generator(query_ilm.unsqueeze(0), init_hidden)

            # TCPGen for slot value generation
            if self.tcpgen and trees is not None:
                ptr_dist, p_gen, treetrack, KBemb = self.forward_tcpgen_inference(self.eos, trees[i], trees[i], next_output[0, -1])
                if self.memnet:
                    next_output = self.decoder(self.memnetlayer(torch.cat([next_output[0, -1], KBemb[0, 0]], dim=-1)))
                else:
                    next_output = torch.softmax(self.decoder(next_output[0, -1]), dim=-1)
                    ptr_gen_complement = ptr_dist[-1:] * p_gen
                    # ilm
                    # ilm_output = torch.softmax(self.decoder(ilm_output[0, -1]), dim=-1)# ** 0.1
                    next_output = (1 - p_gen + ptr_gen_complement) * next_output + ptr_dist[:-1] * p_gen
            else:
                next_output = self.decoder(next_output[0, -1])

            outputtok = torch.max(next_output, dim=0)[1]
            gen_tokens = [outputtok.item()]
            tok_emb = decoder_emb(outputtok)
            while gen_tokens[-1] != self.eos and len(gen_tokens) < maxlen:
                if self.use_gpt_gen:
                    if self.residual:
                        tok_emb = torch.cat([tok_emb, tok_emb.new_zeros(self.dunits-self.embdim)], dim=-1)
                    if gptmodel is not None and self.char_list[outputtok].endswith('▁'):
                        lastword = self.get_wordtext(gen_tokens)
                        current_text = rectext + '\n' + querytext[i] + ' ' + lastword
                        gptemb = gptmodel.forward_inference_gen([current_text])
                        tok_emb = torch.cat([tok_emb, gptemb], dim=-1)
                    else:
                        tok_emb = torch.cat([tok_emb, tok_emb.new_zeros(slotgptemb.size(-1))], dim=-1)
                    # tok_emb = self.slotjointproj(tok_emb)
                if self.jointptrgen and ptr_gen is not None:
                    tok_emb = torch.cat([tok_emb, tok_emb.new_zeros(1)])
                next_output, query_init_hidden = self.generator(tok_emb.unsqueeze(0).unsqueeze(0), query_init_hidden)
                # ilm
                # ilm_output, query_ilm_hidden = self.generator(tok_emb.unsqueeze(0).unsqueeze(0), query_ilm_hidden)
                # TCPGen for slot value generation
                if self.tcpgen and trees is not None:
                    ptr_dist, p_gen, treetrack, KBemb = self.forward_tcpgen_inference(outputtok.item(), treetrack, trees[i], next_output[0, -1])
                    if self.memnet:
                        next_output = self.decoder(self.memnetlayer(torch.cat([next_output[0, -1], KBemb[0, 0]], dim=-1)))
                    else:
                        next_output = torch.softmax(self.decoder(next_output[0, -1]), dim=-1)
                        ptr_gen_complement = ptr_dist[-1:] * p_gen
                        next_output = (1 - p_gen + ptr_gen_complement) * next_output + ptr_dist[:-1] * p_gen
                        # ilm
                        # next_output = torch.log(next_output) - 0.1 * torch.log_softmax(ilm_output, dim=-1)
                    # print(ptr_dist[:-1].sum() * p_gen)
                else:
                    next_output = self.decoder(next_output[0, -1])

                outputtok = torch.max(next_output, dim=0)[1]
                tok_emb = decoder_emb(outputtok)
                gen_tokens.append(outputtok.item())
            slot_output.append(gen_tokens[:-1])
        return slot_output

    def inference_simpletod(self, d_hidden, decoder_emb, gpthidden=None, yseq=[],
                            ptr_gen=None, gptmodel=None, pasttext=(), history=None):
        maxlen = 1024
        rectext, querytext = pasttext
        if isinstance(d_hidden, list):
            d_hidden = torch.cat(d_hidden, dim=0)
        if self.jointrep:
            gpthidden = self.remap_query(gpthidden, [yseq])
            if gpthidden.size(1) > d_hidden.size(0):
                gpthidden = gpthidden[:, :d_hidden.size(0)]
            d_hidden = self.mmproj(torch.cat([d_hidden, gpthidden.squeeze(0)], dim=-1))
        else:
            d_hidden = self.mmproj(d_hidden)

        init_hidden = self.init_hidden(1)
        if self.jointptrgen and ptr_gen is not None:
            # d_hidden = torch.cat([d_hidden, torch.stack(ptr_gen)[:,1:]], dim=-1)
            d_hidden = torch.cat([d_hidden, torch.tensor(ptr_gen).unsqueeze(-1)], dim=-1)
            # ptr_gen = torch.stack(ptr_gen)[:,1:]
        slot_output, next_hidden = self.generator(d_hidden.unsqueeze(0), init_hidden)
        slot_output = []
        outputtok = torch.LongTensor([self.eos])
        tok_emb = decoder_emb(outputtok)[0]
        while (len(slot_output) == 0 or slot_output[-1] != self.eos) and len(slot_output) < maxlen:
            if self.char_list[outputtok].endswith('▁'):
                lastword = self.get_wordtext(slot_output)
                if history is not None:
                    current_text = history[0] + rectext + '\n' + lastword
                else:
                    current_text = rectext + '\n' + lastword
                gptemb = gptmodel.forward_inference_gen([current_text])
                tok_emb = torch.cat([tok_emb, gptemb], dim=-1)
            else:
                tok_emb = torch.cat([tok_emb, tok_emb.new_zeros(gpthidden.size(-1))], dim=-1)
            tok_emb = self.slotjointproj(tok_emb)
            if self.jointptrgen and ptr_gen is not None:
                tok_emb = torch.cat([tok_emb, tok_emb.new_zeros(1)])
            next_output, next_hidden = self.generator(tok_emb.unsqueeze(0).unsqueeze(0), next_hidden)
            next_output = self.decoder(next_output[0, -1])
            outputtok = torch.max(next_output, dim=0)[1]
            tok_emb = decoder_emb(outputtok)
            slot_output.append(outputtok.item())
        return slot_output

    def inference_beam(self, d_hidden, decoder_emb, queryemb, beam=1, gpthidden=None,
                       nonepenalty=0, yseq=[], ptr_gen=None, slothidden=None, slotids=None,
                       gptmodel=None, trees=None, pasttext=()):
        maxlen = max(3, 2 * len(d_hidden))
        rectext, querytext = pasttext
        if isinstance(d_hidden, list):
            d_hidden = torch.cat(d_hidden, dim=0)
        if self.jointrep:
            gpthidden = self.remap_query(gpthidden, [yseq])
            if gpthidden.size(1) > d_hidden.size(0):
                gpthidden = gpthidden[:, :d_hidden.size(0)]
            d_hidden = torch.cat([d_hidden, gpthidden.squeeze(0)], dim=-1)

        init_hidden = self.init_hidden(1)
        if self.jointptrgen and ptr_gen is not None and len(ptr_gen) > 0:
            if isinstance(ptr_gen[0], float):
                ptr_gen = torch.tensor(ptr_gen).unsqueeze(1).repeat(1, len(queryemb))
            else:
                ptr_gen = torch.stack(ptr_gen)[:,1:]
        else:
            slot_output, reset_hidden = self.generator(d_hidden.unsqueeze(0), init_hidden)
        slot_output = []
        for n, query in enumerate(queryemb):
            if self.jointptrgen and ptr_gen is not None:
                _, reset_hidden = self.generator(torch.cat([d_hidden, ptr_gen[:,n:n+1]], dim=-1).unsqueeze(0), init_hidden)
            if self.use_gpt_gen:
                slotgptemb = self.remap_query(slothidden[n:n+1], [slotids[n][1:]], pad_id=0)
                if self.residual:
                    query_pad = query.new_zeros(query.size(0), self.dunits-self.embdim)
                    query = torch.cat([query, query_pad, slotgptemb.squeeze(0)], dim=-1)
                else:
                    query = torch.cat([query, slotgptemb.squeeze(0)], dim=-1)
            if self.jointptrgen and ptr_gen is not None:
                query = torch.cat([query, query.new_zeros(query.size(0), 1)], dim=-1)
            next_output, init_hidden = self.generator(query.unsqueeze(0), reset_hidden)

            # TCPGen for slot value generation
            treetrack = None
            if self.tcpgen and trees is not None:
                ptr_dist, p_gen, treetrack, KBemb = self.forward_tcpgen_inference(self.eos, trees[n], trees[n], next_output[0, -1])
                # if trees[i][0] != {}:
                #     p_gen = max(p_gen, 0.5)
                next_output = torch.softmax(self.decoder(next_output[0, -1]), dim=-1)
                ptr_gen_complement = ptr_dist[-1:] * p_gen
                next_output = (1 - p_gen + ptr_gen_complement) * next_output + ptr_dist[:-1] * p_gen
                next_output = torch.log(next_output)
            else:
                next_output = torch.log_softmax(self.decoder(next_output[0, -1]), dim=-1)

            outputscores, outputtok = torch.topk(next_output, k=beam)
            hyps = [{'yseq':outputtok[i:i+1], 'hidden':init_hidden, 'score': outputscores[i], 'tree': treetrack} for i, idx in enumerate(outputtok)]
            ended_hyps = []
            for i in range(maxlen):
                kept_best_hyps = []
                for hyp in hyps:
                    tok_emb = decoder_emb(hyp['yseq'][-1])
                    if self.use_gpt_gen:
                        if self.residual:
                            tok_emb = torch.cat([tok_emb, tok_emb.new_zeros(self.dunits-self.embdim)], dim=-1)
                        if gptmodel is not None and self.char_list[hyp['yseq'][-1]].endswith('▁'):
                            lastword = self.get_wordtext(hyp['yseq'])
                            current_text = rectext + '\n' + querytext[n] + ' ' + lastword
                            gptemb = gptmodel.forward_inference_gen([current_text])
                            tok_emb = torch.cat([tok_emb, gptemb], dim=-1)
                        else:
                            tok_emb = torch.cat([tok_emb, tok_emb.new_zeros(slotgptemb.size(-1))], dim=-1)
                    if self.jointptrgen and ptr_gen is not None:
                        tok_emb = torch.cat([tok_emb, tok_emb.new_zeros(1)])
                    next_output, beam_hidden = self.generator(tok_emb.unsqueeze(0).unsqueeze(0), hyp['hidden'])
                    # TCPGen for slot value generation
                    if self.tcpgen and trees is not None:
                        ptr_dist, p_gen, treetrack, KBemb = self.forward_tcpgen_inference(hyp['yseq'][-1], hyp['tree'], trees[n], next_output[0, -1])
                        next_output = torch.softmax(self.decoder(next_output[0, -1]), dim=-1)
                        ptr_gen_complement = ptr_dist[-1:] * p_gen
                        next_output = (1 - p_gen + ptr_gen_complement) * next_output + ptr_dist[:-1] * p_gen
                        next_output = torch.log(next_output)
                    else:
                        next_output = torch.log_softmax(self.decoder(next_output[0, -1]), dim=-1)

                    topkscores, topkids = torch.topk(next_output, k=beam)
                    for k, score in enumerate(topkscores):
                        newhyp = {}
                        newhyp['yseq'] = torch.cat([hyp['yseq'], topkids[k:k+1]], dim=0)
                        newhyp['score'] = hyp['score'] + score.item()
                        newhyp['hidden'] = beam_hidden[:]
                        newhyp['tree'] = treetrack
                        kept_best_hyps.append(newhyp)
                    kept_best_hyps = sorted(
                        kept_best_hyps, key=lambda x: x["score"], reverse=True
                    )[:beam]
                hyps = kept_best_hyps
                remained_hyps = []
                for hyp in hyps:
                    if hyp["yseq"][-1].item() == self.eos:
                        if hyp["yseq"][:-1].tolist() == self.nonestr:
                            hyp["score"] -= nonepenalty
                        ended_hyps.append(hyp)
                    else:
                        remained_hyps.append(hyp)
                # end detection
                if self.end_detect(ended_hyps, i):
                    # print("end detected at {}".format(i))
                    break
                hyps = remained_hyps
                if len(hyps) <= 0:
                    # print("no hypothesis. Finish decoding.")
                    break
            nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"]/len(x["yseq"]), reverse=True)[0]
            slot_output.append(nbest_hyps['yseq'].tolist()[:-1])
            # slot_output.append(nbest_hyps)
        return slot_output

    def end_detect(self, ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
        """End detection.

        described in Eq. (50) of S. Watanabe et al
        "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

        :param ended_hyps:
        :param i:
        :param M:
        :param D_end:
        :return:
        """
        if len(ended_hyps) == 0:
            return False
        count = 0
        best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
        for m in range(M):
            # get ended_hyps with their length is i - m
            hyp_length = i - m
            hyps_same_length = [x for x in ended_hyps if len(x["yseq"]) == hyp_length]
            if len(hyps_same_length) > 0:
                best_hyp_same_length = sorted(
                    hyps_same_length, key=lambda x: x["score"], reverse=True
                )[0]
                if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                    count += 1

        if count == M:
            return True
        else:
            return False
