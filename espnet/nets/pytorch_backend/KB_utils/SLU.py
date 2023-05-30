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
        self.ontology = None
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
            uncovered = [slot for slot in self.ids2slot[1:] if self.slot2ids[slot] not in slotlist]
            if self.ontology is not None:
                uncovered = [slot for slot in uncovered if self.ontology[slot] != []]
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
        # if self.wordlevel:
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
                 classpost=False, postfactor=0.0, attndim=256):
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
            if postfactor >= 1.0:
                self.classprobweight = torch.nn.Linear(jointdim+(jointdim if jointrep else 0), 1)
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
                classpost=None, KBemb=[]):
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
        slot_hidden = self.dropout(slot_hidden)
        slot_logits = self.slothead(slot_hidden)
        if ptr_gen is not None and self.classpost and classpost != []:
            slot_logits = torch.softmax(slot_logits, dim=-1)
            ptr_gen = ptr_gen.unsqueeze(-1)
            if self.postfactor >= 1.0 and KBemb != []:
                # classprobweight = torch.sigmoid(self.classprobweight(torch.cat([slot_hidden, KBemb], dim=-1)))
                classprobweight = torch.sigmoid(self.classprobweight(slot_hidden))
                # ptrmask = ptr_gen != 0
                # classprobweight = ptrmask * classprobweight
                slot_logits = ptr_gen * classprobweight * classpost + (1 - ptr_gen * classprobweight) * slot_logits
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

    def inference(self, d_hidden, y_hyp, mmemb=None, ptr_gen=None, topn=10, clspost=None, KBemb=[]):
        d_hidden = torch.cat(d_hidden, dim=0)
        if ptr_gen is not None:
            ptr_gen = torch.clamp(to_device(self, torch.tensor(ptr_gen).unsqueeze(1)), max=1.0)
            if self.jointptrgen:
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
                lasthidden = slot_hidden[-1]
            if self.jointrep:
                slot_hidden = torch.cat([slot_hidden, roberta_emb], dim=-1)
                lasthidden = torch.cat([lasthidden, roberta_pool[0]], dim=-1)
        # intent_logits = self.intenthead(self.intentproj(lasthidden))
        intent_logits = self.intenthead(lasthidden)
        intent_pred = intent_logits.max(dim=-1)[1].item()
        slot_logits = self.slothead(slot_hidden)
        if ptr_gen is not None and self.classpost and clspost != []:
            classpost = torch.stack(clspost)
            slotdist = torch.softmax(slot_logits, dim=-1)
            if self.postfactor >= 1.0 and KBemb != []:
                # KBemb = torch.stack(KBemb).squeeze(1)
                # classprobweight = torch.sigmoid(self.classprobweight(torch.cat([slot_hidden, KBemb], dim=-1)))
                classprobweight = torch.sigmoid(self.classprobweight(slot_hidden))
                print(classprobweight)
                ptr_gen = ptr_gen * classprobweight # self.postfactor
                slot_logits = ptr_gen * classpost + (1 - ptr_gen) * slotdist
            else:
                slot_logits = ptr_gen * self.postfactor * classpost + (1 - ptr_gen * self.postfactor) * slotdist
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
        # for i, slotlist in enumerate(topnslots):
        #     for slot in slots[i].tolist():
        #         if slot != 0 and slot != -1 and slot not in slotlist:
        #             slotlist.append(slot)
        return topnslots
