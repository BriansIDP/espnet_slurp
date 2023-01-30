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

from espnet.lm.lm_utils import make_lexical_tree


class KBmeeting(object):
    def __init__(self, vocabulary, meetingpath, charlist, bpe=False):
        """Meeting-wise KB in decoder
        """
        self.meetingdict = {}
        self.meetingdict_sym = {}
        self.meetingmask = {}
        self.meetinglextree = {}
        self.chardict = {}
        self.charlist = charlist
        self.bpe = bpe
        for i, char in enumerate(charlist):
            self.chardict[char] = i

        self.maxlen = 0
        self.unkidx = vocabulary.get_idx('<unk>')
        for filename in os.listdir(meetingpath):
            worddict, wordlist = {}, []
            with open(os.path.join(meetingpath, filename), encoding='utf-8') as fin:
                for word in fin:
                    word = tuple(word.split()) if bpe else word.strip()
                    worddict[word] = len(wordlist) + 1
                    wordlist.append(word)
            self.meetingdict[filename] = vocabulary.get_ids(wordlist, oov_sym='<blank>')
            self.meetinglextree[filename] = make_lexical_tree(worddict, self.chardict, -1)
            self.maxlen = len(wordlist) if len(wordlist) > self.maxlen else self.maxlen
        # pad meeting wordlist
        for meeting, wordlist in self.meetingdict.items():
            self.meetingdict_sym[meeting] = vocabulary.get_syms(self.meetingdict[meeting])
            self.meetingdict[meeting] = wordlist + [self.unkidx] * (self.maxlen - len(wordlist) + 1)
            self.meetingmask[meeting] = [0] * (len(wordlist)) + [1] * (self.maxlen - len(wordlist)) + [0]
        self.unkidx = self.maxlen
        self.maxlen = self.maxlen + 1
        self.vocab = vocabulary

    def get_meeting_KB(self, meetinglist, bsize=0):
        KBlist = torch.LongTensor([self.meetingdict[meeting] for meeting in meetinglist])
        KBmask = torch.Tensor([self.meetingmask[meeting] for meeting in meetinglist])
        return KBlist, KBmask.byte(), meetinglist


class KBmeetingTrainContext(object):
    def __init__(self, vocabulary, meetingpath, charlist, bpe=False, maxlen=800, DBdrop=0.0):
        """Meeting-wise KB in decoder
        """
        self.meetingdict = {}
        self.meetingdict_sym = {}
        self.meetingmask = {}
        self.meetinglextree = {}
        self.chardict = {}
        self.charlist = charlist
        self.rarewords = []
        self.bpe = bpe
        for i, char in enumerate(charlist):
            self.chardict[char] = i

        self.maxlen = maxlen
        self.unkidx = maxlen - 1
        for filename in os.listdir(meetingpath):
            if filename == 'rarewords.txt':
                with open(os.path.join(meetingpath, filename), encoding='utf-8') as fin:
                    for line in fin:
                        self.rarewords.append(tuple(line.split()))
            else:
                wordlist = set()
                with open(os.path.join(meetingpath, filename), encoding='utf-8') as fin:
                    for word in fin:
                        wordlist.add(tuple(word.split()))
                    assert len(wordlist) <= self.maxlen
                    self.meetingdict[filename] = wordlist
        self.vocab = vocabulary
        self.DBdrop = DBdrop
        self.curriculum = False
        self.fullepoch = 0

    def get_meeting_KB(self, meetings, bsize):
        KBlist = []
        KBmask = torch.Tensor([[0] * self.maxlen]*bsize)
        KBlextrees = []
        for meeting in meetings:
            uttwordlist = self.meetingdict[meeting]
            if self.DBdrop > 0:
                uttwordlist = {word for word in uttwordlist if random.random() > self.DBdrop}
            pre_sampled_words = random.sample(self.rarewords, k=self.maxlen)
            sampled_words = list(uttwordlist)
            count = 0
            while len(sampled_words) < self.maxlen:
                if pre_sampled_words[count] not in uttwordlist:
                    sampled_words.append(pre_sampled_words[count])
                count += 1
            sampled_words = list(uttwordlist) + sampled_words[:self.maxlen-len(uttwordlist)]
            assert len(sampled_words) == self.maxlen
            uttKB = sorted(sampled_words)
            worddict = {word:i+1 for i, word in enumerate(uttKB)}
            lextree = make_lexical_tree(worddict, self.chardict, -1)
            # KBlist = torch.LongTensor([wordlist]*bsize)
            KBlextrees.append(lextree)
        return KBlist, KBmask.byte(), KBlextrees


class KBmeetingTrain(object):
    def __init__(self, vocabulary, meetingpath, charlist, bpe=False, maxlen=800, DBdrop=0.0,
                 curriculum=False, fullepoch=0, unigram=''):
        """Meeting-wise KB in decoder
        """
        self.meetingdict = {}
        self.meetingdict_sym = {}
        self.meetingmask = {}
        self.meetinglextree = {}
        self.chardict = {}
        self.charlist = charlist
        self.bpe = bpe

        # load in unigram probability
        self.unigram = unigram
        self.unigram_count = {}
        if self.unigram != '':
            with open(unigram, encoding='utf-8') as fin:
                for line in fin:
                    word = tuple(line.split()[:-1])
                    freq = int(line.split()[-1])
                    self.unigram_count[word] = freq

        # get char dict for lextree contruction
        for i, char in enumerate(charlist):
            self.chardict[char] = i

        self.rarewords = []
        self.rareword_dict = {}
        self.rarewords_word = {}
        # load in the rare word list
        with open(meetingpath, encoding='utf-8') as fin:
            for line in fin:
                self.rareword_dict[tuple(line.split())] = len(self.rarewords)
                self.rarewords_word[''.join(line.split()).strip('▁').replace('▁', ' ')] = len(self.rarewords)
                self.rarewords.append(tuple(line.split()))

        self.unkidx = maxlen - 1
        self.vocab = vocabulary
        self.DBdrop = DBdrop
        self.epoch = 0
        self.curriculum = curriculum
        self.fullepoch = fullepoch
        self.get_full_KB()

    def get_meeting_KB(self, uttwordlist, bsize):
        DBdrop = self.DBdrop
        # true_list = uttwordlist
        # removing oracle words from KB to regularise
        if self.DBdrop > 0:
            if self.DBdrop >= 1 and self.unigram_count != {}:
                uttwordlist = [word for word in uttwordlist if random.random() < (
                    1 - (self.unigram_count[word] / self.DBdrop) if word in self.unigram_count else 1)]
            else:
                uttwordlist = [word for word in uttwordlist if random.random() > self.DBdrop]
        # for MBR
        true_list = uttwordlist
        pre_sampled_words = random.sample(self.rarewords, k=self.maxlen)
        sampled_words = []
        for word in pre_sampled_words:
            if word not in uttwordlist:
                sampled_words.append(word)
        maxsample_len = self.maxlen
        sampled_words = list(uttwordlist) + sampled_words[:maxsample_len-len(uttwordlist)]
        assert len(sampled_words) == maxsample_len
        uttKB = sorted(sampled_words)
        worddict = {word:i+1 for i, word in enumerate(uttKB)}
        lextree = make_lexical_tree(worddict, self.chardict, -1)
        wordlist = [] # self.vocab.get_ids(uttKB, oov_sym='<blank>')
        # wordmasks = [0] * self.maxlen
        KBlist = torch.LongTensor([wordlist]*bsize)
        # KBmask = torch.Tensor([wordmasks]*bsize)
        KBlextrees = [lextree]*bsize
        return KBlist, true_list, KBlextrees

    def get_full_KB(self):
        worddict = {word:i+1 for i, word in enumerate(self.rarewords)}
        self.full_lextree = [make_lexical_tree(worddict, self.chardict, -1)]
        self.full_wordlist = torch.LongTensor([self.vocab.get_ids(self.rarewords, oov_sym='<blank>')])

    def get_tree_from_classes(self, classfile, classorder):
        with open(classfile) as fin:
            self.classdict = json.load(fin)
        with open(classorder) as fin:
            self.classorder = [line.strip() for line in fin]
        self.classtrees = []
        for cls in self.classorder:
            bpe_ids = [self.rarewords_word[word] for word in self.classdict[cls]]
            self.classtrees.append(self.get_tree_from_inds(bpe_ids))

    def get_classed_trees(self, classes):
        classed_trees = []
        for cls in classes:
            classed_trees.append(self.classtrees[cls.item()])
        return classed_trees

    def get_tree_from_inds(self, inds, extra=None, true_list=[]):
        autoKBwords = set()
        for index in inds:
            if self.rarewords[index] not in true_list:
                autoKBwords.add(self.rarewords[index])
        for word in true_list:
            if self.DBdrop >= 0 and random.random() > self.DBdrop:
                autoKBwords.add(word)
        if extra is not None:
            for word in extra:
                if word not in autoKBwords:
                    autoKBwords.add(word)
        uttKB = sorted(list(autoKBwords))
        worddict = {word:i+1 for i, word in enumerate(uttKB)}
        lextree = make_lexical_tree(worddict, self.chardict, -1)
        return lextree


class Vocabulary(object):
    def __init__(self, dictfile, bpe=False):
        self.sym2idx = {}
        self.idx2sym = []
        self.maxwordlen = 0
        self.bpe = bpe
        with open(dictfile, encoding='utf-8') as fin:
            for i, line in enumerate(fin):
                if bpe:
                    word = tuple(line.split()[1:])
                    self.sym2idx[word] = i
                    self.idx2sym.append(word)
                else:
                    word, ind = line.split()
                    self.sym2idx[word] = int(ind)
                    self.idx2sym.append(word)
                if len(word) > self.maxwordlen:
                    self.maxwordlen = len(word)
        if '<eos>' not in self.sym2idx:
            self.sym2idx['<eos>'] = len(self.idx2sym)
            self.idx2sym.append('<eos>')
        if '<unk>' not in self.sym2idx:
            self.sym2idx['<unk>'] = len(self.idx2sym)
            self.idx2sym.append('<unk>')
        if '<blank>' not in self.sym2idx:
            self.sym2idx['<blank>'] = len(self.idx2sym)
            self.idx2sym.append('<blank>')
        self.ntokens = len(self.idx2sym)

    def get_ids(self, textlist, oov_sym='<unk>'):
        return_list = []
        for word in textlist:
            return_list.append(self.get_idx(word, oov_sym=oov_sym))
        return return_list

    def get_idx(self, word, oov_sym='<unk>'):
        if word not in self.sym2idx:
            return self.sym2idx[oov_sym]
        else:
            return self.sym2idx[word]

    def get_syms(self, ids):
        return [self.idx2sym[i] for i in ids]
