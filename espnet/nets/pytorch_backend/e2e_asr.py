# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech recognition model (pytorch)."""

import argparse
from itertools import groupby
import logging
import math
import os
import random
import time

import chainer
from chainer import reporter
import editdistance
import numpy as np
import six
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.frontends.feature_transform import (
    feature_transform_for,  # noqa: H301
)
from espnet.nets.pytorch_backend.frontends.frontend import frontend_for
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.argument import (
    add_arguments_rnn_encoder_common,  # noqa: H301
    add_arguments_rnn_decoder_common,  # noqa: H301
    add_arguments_rnn_attention_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
# gs534 - add KB classes
from espnet.nets.pytorch_backend.KB_utils.KB import KBmeeting, KBmeetingTrain
from espnet.nets.pytorch_backend.KB_utils.KB import KBmeetingTrainContext, Vocabulary
from espnet.nets.pytorch_backend.KB_utils.SLU import SLUGenutils, SLUGenNet
from espnet.nets.pytorch_backend.KB_utils.wer import editDistance, getStepList
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args
# Modality matching
from espnet.nets.pytorch_backend.modality.roberta import Roberta_encoder, GPT2_encoder

CTC_LOSS_THRESHOLD = 10000
random.seed(1)


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss, mbrloss,
               mmloss, sluloss, slotacc, intentacc, copyloss):
        """Report at every step."""
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"loss_MBR": mbrloss}, self)
        reporter.report({"loss_Modality": mmloss}, self)
        reporter.report({"loss_slu": sluloss}, self)
        reporter.report({"loss_copy": copyloss}, self)
        reporter.report({"slotacc": slotacc}, self)
        reporter.report({"intentacc": intentacc}, self)
        reporter.report({"acc": acc}, self)
        # reporter.report({"cer_ctc": cer_ctc}, self)
        # reporter.report({"cer": cer}, self)
        # reporter.report({"wer": wer}, self)
        # logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.encoder_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        """Add arguments for the encoder."""
        group = parser.add_argument_group("E2E encoder setting")
        group = add_arguments_rnn_encoder_common(group)
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for the attention."""
        group = parser.add_argument_group("E2E attention setting")
        group = add_arguments_rnn_attention_common(group)
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for the decoder."""
        group = parser.add_argument_group("E2E decoder setting")
        group = add_arguments_rnn_decoder_common(group)
        return parser

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        if isinstance(self.enc, torch.nn.ModuleList):
            return self.enc[0].conv_subsampling_factor * int(np.prod(self.subsample))
        else:
            return self.enc.conv_subsampling_factor * int(np.prod(self.subsample))

    def __init__(self, idim, odim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.etype = args.etype
        self.verbose = args.verbose
        # NOTE: for self.build method
        args.char_list = getattr(args, "char_list", None)
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # gs534 - word vocab
        bpe = ('<space>' not in self.char_list) # hack here for bpe flag
        self.vocabulary = Vocabulary(args.dictfile, bpe) if getattr(args, 'dictfile', '') != '' else None

        # gs534 - create lexicon tree
        lextree = None
        self.meeting_KB = None
        self.n_KBs = getattr(args, 'dynamicKBs', 0)
        self.DBdrop = getattr(args, 'DBdrop', 0.0)
        self.curriculum = getattr(args, 'curriculum', False)
        self.fullepoch = getattr(args, 'fullepoch', 0)
        self.unigram = getattr(args, 'unigram_file', '')
        self.KBminlen = getattr(args, 'KBminlen', args.KBmaxlen)
        self.mmfactor = getattr(args, 'mmfactor', 0.0)
        self.domm = getattr(args, 'modalitymatch', False)
        self.classpost = getattr(args, 'classpost', False)
        self.classpostfactor = getattr(args, 'classpostfactor', 0.0)
        if getattr(args, 'meetingKB', False) and getattr(args, 'meetingpath', '') != '':
            if self.n_KBs == 0:
                if args.randomKBsample:
                    self.meeting_KB = KBmeetingTrain(self.vocabulary, args.meetingpath, args.char_list, bpe,
                                                    maxlen=args.KBmaxlen, DBdrop=self.DBdrop, curriculum=self.curriculum,
                                                    fullepoch=self.fullepoch, unigram=self.unigram, minlen=self.KBminlen)
                else:
                    self.meeting_KB = KBmeeting(self.vocabulary, args.meetingpath, args.char_list, bpe)
            else:
                self.meeting_KB = KBmeetingTrainContext(self.vocabulary, args.meetingpath, args.char_list, bpe,
                                                        maxlen=args.KBmaxlen, DBdrop=self.DBdrop)

        # subsample info
        self.subsample = get_subsample(args, mode="asr", arch="rnn")

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(
                odim, args.lsm_type, transcript=args.train_json
            )
        else:
            labeldist = None

        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels
        else:
            self.frontend = None

        # encoder
        self.enc = encoder_for(args, idim, self.subsample)
        # ctc
        self.ctc = ctc_for(args, odim)
        # attention
        self.att = att_for(args)
        # decoder
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist,
            meetingKB=self.meeting_KB[0] if isinstance(self.meeting_KB, list) else self.meeting_KB)

        # gs534 - SLU related
        self.jointrep = getattr(args, 'jointrep', False)
        self.doslu = getattr(args, 'doslu', False)
        self.useslotKB = getattr(args, 'slotKB', False)
        self.slusche = getattr(args, 'slusche', 0)
        self.use_gpt_gen = getattr(args, 'usegptgen', False)
        self.jointptrgen = getattr(args, 'jointptrgen', False)
        self.classentity = getattr(args, 'classentity', False)
        self.fullslottext = getattr(args, 'fullslottext', False)
        self.simpletod = getattr(args, 'simpletod', False)
        self.copylossfac = getattr(args, 'copylossfac', 0)
        self.gethistory = getattr(args, 'gethistory', False)
        self.topn = min(getattr(args, 'topnslot', 1), args.ndistractors)
        preLMdim = getattr(args, 'robertadim', 768) if self.jointrep else 0
        self.plmmask = getattr(args, 'robertamask', 0.0)
        self.freeze_mod = args.freeze_mods
        self.usetcpgen = getattr(args, 'slottcpgen', False)
        self.usememnet = getattr(args, 'memnet', False) 
        self.ndistr = getattr(args, 'ndistractors', 10)
        if self.doslu:
            self.sluproc = SLUGenutils(args.slotfile, args.connection, self.eos, self.char_list[:], self.ndistr,
                                       ontofile=getattr(args, 'ontology', None), simpletod=self.simpletod)
            self.outputunits = args.dunits + args.eprojs if args.context_residual else args.dunits
            self.slunet = SLUGenNet(self.outputunits, self.sluproc.nslots, args.slotfactor, args.dunits,
                                    self.sluproc.char_list, preLMdim, self.domm, self.jointrep, self.usetcpgen,
                                    attndim=getattr(self.dec, 'attn_dim', 0), embsize=getattr(self.dec, 'tree_hid', 0),
                                    use_gpt_gen=self.use_gpt_gen, connector=self.sluproc.connector,
                                    jointptrgen=self.jointptrgen, nonestr=self.sluproc.none, history=self.gethistory,
                                    copylossfac=self.copylossfac, memnet=self.usememnet)
            if self.usetcpgen:
                self.slunet.embed.weight = self.dec.embed.weight
                self.slunet.ooKBemb.weight = self.dec.ooKBemb.weight

        # weight initialization
        self.init_from = getattr(args, 'init_full_model', None)
        self.init_like_chainer()

        # Load Roberta model
        if self.domm or self.jointrep:
            self.pretrained_LM = GPT2_encoder(pooling=args.pooling, loadfrom=getattr(args, 'init_preLM', None),
                                              gen_with_gpt=self.use_gpt_gen)

        # options for beam search
        if args.report_cer or args.report_wer:
            recog_args = {
                "beam_size": args.beam_size,
                "penalty": args.penalty,
                "ctc_weight": args.ctc_weight,
                "maxlenratio": args.maxlenratio,
                "minlenratio": args.minlenratio,
                "lm_weight": args.lm_weight,
                "rnnlm": args.rnnlm,
                "nbest": args.nbest,
                "space": args.sym_space,
                "blank": args.sym_blank,
            }

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

        # MBR related
        self.use_mbrloss = getattr(args, 'mbrloss', False)
        self.mbrbeam = getattr(args, 'mbrbeam', 1)
        self.mbrnbest = getattr(args, 'mbrnbest', 1)
        self.mbrlambda = getattr(args, 'mbrlambda', 0.0)
        self.useGT = getattr(args, 'mbrGT', False)
        self.rareweight = getattr(args, 'mbrrareweight', 0.0)
        self.mwe_factor = getattr(args, 'mweweight', 1.0)
        self.cfm_mbr_start = getattr(args, 'cfm_mbr_start', 0)
        self.use_wp_errors = getattr(args, 'use_wp_errors', False)

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for i in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[i].bias_ih)

    def load_from(self):
        model_init = torch.load(self.init_from, map_location=lambda storage, loc: storage)
        model_init = model_init.state_dict() if not isinstance(model_init, dict) else model_init
    
    def forward_frontend_and_encoder(self, xs_pad, ilens):
        """Forward front-end and encoder."""
        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, _ = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)
        return hs_pad, hlens

    def get_oracle_ptrgen_backup(self, ys_pad, wlists):
        oracle_gens = []
        wordids = []
        for word in wlists:
            wordids.append([self.char_list.index(charstr) if charstr in self.char_list else 0 for charstr in word])
        for batchid, ys in enumerate(ys_pad):
            new_oracle = []
            cumword = []
            for i, charidx in enumerate(ys):
                cumword.append(charidx)
                if charidx != -1 and self.char_list[charidx].endswith('▁'):
                    if cumword in wordids:
                        new_oracle.extend([1.0] * len(cumword))
                    else:
                        new_oracle.extend([0.0] * len(cumword))
                    cumword = []
            if cumword != []:
                new_oracle.extend([0.0] * len(cumword))
            new_oracle.append(0.0)
            oracle_gens.append(new_oracle)
        oracle_gens = torch.tensor(oracle_gens).to(ys_pad.device)
        return oracle_gens

    def get_oracle_ptrgen(self, ys_pad, wlists):
        oracle_gens = []
        wordids = []
        for batchid, ys in enumerate(ys_pad):
            ys = ys.tolist()
            new_oracle = [0.0] * (len(ys) + 1)
            for word in wlists[batchid]:
                wordlist = [self.char_list.index(charstr) if charstr in self.char_list else 0 for charstr in word]
                for pos, charidx in enumerate(ys):
                    if ys[pos:pos+len(wordlist)] == wordlist:
                        new_oracle[pos:pos+len(wordlist)] = [1.0] * len(wordlist)
            oracle_gens.append(new_oracle)
        oracle_gens = torch.tensor(oracle_gens).to(ys_pad.device)
        return oracle_gens

    def forward(self, xs_pad, ilens, ys_pad, meetings=None, orig_text=None, intents=None, slots=None,
                history=None):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 0. get slu labels
        if self.doslu:
            if self.freeze_mod is not None and 'enc' in self.freeze_mod:
                # stop updating batchnorm layer running stats
                self.enc.encoders.eval() 
            slotlabel, slotmask, slotmap, slottext, slotlist, clsprobmask = self.sluproc.get_slot_labels(
                slots, ys_pad, self.fullslottext, self.topn)
            slotlabel = to_device(self, slotlabel)
            slotmask = to_device(self, slotmask)
            slotmap = to_device(self, slotmap)
            # clsprobmask = to_device(self, torch.tensor(clsprobmask))
            ontology_trees = None
        # 1. forward front end & encoder
        hs_pad, hlens = self.forward_frontend_and_encoder(xs_pad, ilens)

        # 2. CTC loss
        if self.mtlalpha == 0:
            self.loss_ctc = None
        else:
            self.loss_ctc = self.ctc(hs_pad, hlens, ys_pad)

        # 3. attention loss
        if self.mtlalpha == 1:
            self.loss_att, acc = None, None
        else:
            # gs534 - meeting KB
            meeting_info = None
            if self.meeting_KB is not None and not self.useslotKB:
                meeting_info = self.meeting_KB.get_meeting_KB(meetings, ilens.size(0))

            # Modality Matching
            mmembs = None
            newhidden = None
            slothidden = None
            tcpgen_trees = None
            copylabels = None
            if self.doslu:
                if self.domm or self.jointrep:
                    newhidden, slothidden = self.pretrained_LM(orig_text, slotlabel=slottext, slotmap=slotmap,
                                                               training=self.jointrep, history=history)
                mmembs = self.dec.embed(slotlabel)

                if self.useslotKB:
                    wlists, slotwlists = self.sluproc.get_wlist_from_slots(slotlist, self.DBdrop, self.topn)
                    meeting_info = self.meeting_KB.get_slot_KB(wlists, entity=self.classentity)
                    tcpgen_trees = self.meeting_KB.get_slot_sep_KB(slotwlists, entity=self.classentity)

            self.loss_att, acc, _, doutput, ptr_gen, classpost = self.dec(hs_pad, hlens, ys_pad, meeting_info=meeting_info,
                                                                          slottrees=tcpgen_trees)
            if self.doslu and meeting_info is not None:
                if self.dec.epoch <= self.dec.PtrSche and self.jointptrgen and not self.usetcpgen and not self.useslotKB:
                    copywlist = [meetings] * ys_pad.size(0)
                else:
                    copywlist = meeting_info[1] if self.useslotKB else meeting_info[0]
                copylabels, appeared_words, words_per_slot = self.sluproc.get_copy_labels(copywlist, slottext, slotlabel, slotmap, slotmask,
                                                                                          entity=self.classentity)
                if self.dec.epoch <= self.dec.PtrSche and self.jointptrgen:
                    ptr_gen = self.get_oracle_ptrgen(ys_pad[slotmap], words_per_slot)

            # SLU loss
            self.sluloss = 0
            self.loss_mm = 0
            self.slotacc, self.intentacc, self.copyloss = 0, 0, 0
            maskfactor = 1.0 if self.gethistory and self.dec.epoch < 0 else self.plmmask
            if self.doslu:
                if self.usetcpgen and meeting_info is not None and self.dec.epoch >= self.slusche:
                    slotbiasinglists = self.meeting_KB.get_slotbiasing(copywlist, appeared_words)
                    if self.slunet.memnet:
                        ontology_trees = slotbiasinglists
                    else:
                        ontology_trees = []
                        for i, slotwordlist in enumerate(slotbiasinglists):
                            lextree = self.meeting_KB.get_tree(slotwordlist)
                            if lextree[0] != {}:
                                self.dec.encode_tree(lextree)
                            ontology_trees.append(lextree)
                if self.useslotKB and self.dec.epoch > self.dec.PtrSche:
                    classpost = torch.cat(
                        [classpost, classpost.new_zeros(classpost.size(0), classpost.size(1), self.ndistr-classpost.size(2))], dim=-1)
                    ptr_gen = classpost.transpose(1, 2).contiguous().view(-1, classpost.size(1))
                self.sluloss, self.slotacc, self.copyloss = self.slunet(doutput, ys_pad, slotlabel, slotmask, slotmap, mmembs, gpthidden=newhidden,
                                                            trees=ontology_trees, slothidden=slothidden, ptr_gen=ptr_gen,
                                                            maskfactor=maskfactor, copylabel=copylabels)

            # gs534 - MBR training
            if self.use_mbrloss and ys_pad.size(1) > 0 and self.dec.epoch >= self.cfm_mbr_start:
                #  set maximum decoding length to avoid running out of mem
                y_maxlen = min(int(ys_pad.size(1) * 1.5), hlens.max(), 210)
                # print(ys_pad.size())
                nbest = self.dec.batch_decode_nbest(hs_pad, hlens, beam=self.mbrbeam,
                                                    nbest=self.mbrnbest, maxlen=y_maxlen, meeting_info=meeting_info)
                lossGT = lossGT.view(ys_pad.size(0), -1).sum(1) if self.useGT else None
                if self.meeting_KB is not None:
                    self.mbrloss, self.mbrloss_unnorm = self.get_KBmbr_loss(ys_pad, nbest, meeting_info[1], loss_sep=lossGT)
                else:
                    self.mbrloss, self.mbrloss_unnorm = self.get_mbr_loss(ys_pad, nbest, loss_sep=lossGT)
                self.loss_att = self.mbrlambda * self.loss_att + self.mbrloss
            else:
                self.mbrloss_unnorm = 0

        self.acc = acc

        # 4. compute cer without beam search
        if self.mtlalpha == 0 or self.char_list is None:
            cer_ctc = None
        else:
            cers = []

            y_hats = self.ctc.argmax(hs_pad).data
            for i, y in enumerate(y_hats):
                y_hat = [x[0] for x in groupby(y)]
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [
                    self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                ]
                seq_hat_text = "".join(seq_hat).replace(self.space, " ")
                seq_hat_text = seq_hat_text.replace(self.blank, "")
                seq_true_text = "".join(seq_true).replace(self.space, " ")

                hyp_chars = seq_hat_text.replace(" ", "")
                ref_chars = seq_true_text.replace(" ", "")
                if len(ref_chars) > 0:
                    cers.append(
                        editdistance.eval(hyp_chars, ref_chars) / len(ref_chars)
                    )

            cer_ctc = sum(cers) / len(cers) if cers else None

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(hs_pad).data
            else:
                lpz = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad,
                torch.tensor(hlens),
                lpz,
                self.recog_args,
                self.char_list,
                self.rnnlm,
            )
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]["yseq"][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [
                    self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                ]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, " ")
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, "")
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, " ")

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))
                hyp_chars = seq_hat_text.replace(" ", "")
                ref_chars = seq_true_text.replace(" ", "")
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

            wer = (
                0.0
                if not self.report_wer
                else float(sum(word_eds)) / sum(word_ref_lens)
            )
            cer = (
                0.0
                if not self.report_cer
                else float(sum(char_eds)) / sum(char_ref_lens)
            )

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = self.loss_att + self.sluloss
            loss_att_data = float(self.loss_att)
            loss_mbr_data = float(self.mbrloss_unnorm)
            loss_mm_data = float(self.loss_mm)
            loss_slu_data = float(self.sluloss)
            loss_copy_data = float(self.copyloss)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = self.loss_ctc
            loss_att_data = None
            loss_ctc_data = float(self.loss_ctc)
        else:
            self.loss = alpha * self.loss_ctc + (1 - alpha) * self.loss_att
            loss_att_data = float(self.loss_att)
            loss_ctc_data = float(self.loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, acc, cer_ctc, cer, wer, loss_data, loss_mbr_data,
                loss_mm_data, loss_slu_data, self.slotacc, self.intentacc, loss_copy_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.dec, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = np.array([x.shape[0]])

        # subsample frame
        x = x[:: self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        hs, _ = self.forward_frontend_and_encoder(hs, ilens)
        return hs.squeeze(0)

    def recognize(self, x, recog_args, char_list, rnnlm=None, meetings=None,
                  best_fusion=None, estlm=None, estlm_factor=0.0, best_est=None, bhist=[],
                  sel_lm=None, prev_hid=None, slotlist=[], oracletext=None, unigram=None, history=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs = self.encode(x).unsqueeze(0)
        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs)[0]
        else:
            lpz = None

        # Get KB info
        meeting_info = None
        classmask = None
        classmap = None
        if self.meeting_KB is not None and not recog_args.select and not recog_args.slotlist:
            meeting_info = self.meeting_KB.get_meeting_KB(meetings, 1)
        elif self.meeting_KB is not None and not recog_args.select and recog_args.slotlist:
            if slotlist != []:
                meeting_info = (self.meeting_KB.full_wordlist, None,
                                [self.meeting_KB.get_classed_trees(slotlist[:recog_args.topk], named=True)])
                classmap = []
                for slot in slotlist:
                    mapvec = [0 for i in range(len(self.sluproc.slotorder)+1)]
                    mapvec[self.sluproc.slot2ids[slot]+1] = 1
                    classmap.append(mapvec)
                classmap = to_device(self, torch.Tensor(classmap))
            else:
                meeting_info = []
            classmask = torch.tensor([0 if tree[0] != {} else 1 for tree in self.meeting_KB.classtrees])
        elif self.meeting_KB is not None and recog_args.select:
            meeting_info = (self.meeting_KB.full_wordlist, meetings,
                self.meeting_KB.full_lextree)
            classmask = torch.tensor([0 if tree[0] != {} else 1 for tree in self.meeting_KB.classtrees])

        # 2. Decoder
        # decode the first utterance
        y, best_fusion, best_est, best_hid = self.dec.recognize_beam(hs[0], lpz, recog_args, char_list, rnnlm,
                                                           meeting_info=meeting_info,
                                                           ranking_norm=recog_args.ranking_norm,
                                                           fusion_lm_hidden=best_fusion, 
                                                           estlm=estlm, estlm_factor=estlm_factor,
                                                           estlm_hid=best_est, dynamic_disc=recog_args.dynamic_disc,
                                                           sel_lm=sel_lm, topk=recog_args.topk, prev_hid=prev_hid,
                                                           classlm=recog_args.classlm, slotlist=recog_args.slotlist,
                                                           unigram=unigram, classmask=classmask, fixedclassmap=classmap)
        # oracle text
        if oracletext is not None:
            hlens = [hs.size(1)]
            ys_pad = torch.LongTensor(oracletext).unsqueeze(0)
            self.loss_att, acc, _, doutput, ptr_gen, classpost = self.dec(hs, hlens, ys_pad, meeting_info=meeting_info)
            y[0]['hidden_states'] = doutput.squeeze(0)
            y[0]['yseq'] = [self.sos] + oracletext + [self.eos]

        # SLU prediction
        laststate = []
        if self.doslu:
            with torch.no_grad():
                orig_text = ''.join([self.char_list[idx] for idx in y[0]['yseq'][1:-1]]).strip('▁').replace('▁', ' ')
                if self.jointrep:
                    history = [history] if history is not None else history
                    newhidden, slothidden = self.pretrained_LM([orig_text.lower()], slotlabel=self.sluproc.slotquerytext,
                                                               slotmap=[0]*len(self.sluproc.slotquerytext), history=history)
                    mmembs = []
                    for i, query in enumerate(self.sluproc.slotqueries):
                        mmembs.append(self.dec.embed(to_device(self, query)))
                else:
                    mmembs = []
                    newhidden = None
                    for query in self.sluproc.slotqueries:
                        mmembs.append(self.dec.embed(to_device(self, query)))

                pasttext = history[0]+orig_text.lower() if history is not None else orig_text.lower()
                # Encode second tree
                if self.slunet.tcpgen and self.meeting_KB is not None:
                    trees = []
                    for k, wlist in enumerate(y[0]['entities']):
                        if bhist != [] and bhist[k] != []:
                            for word in bhist[k]:
                                if word not in wlist:
                                    wlist.append(word)
                        trees.append(self.meeting_KB.get_tree(wlist))
                    # else:
                    #     tree = self.meeting_KB.get_tree(meeting_info[0][0])
                    #     trees = [tree for k in range(self.slunet.nslots)]
                    for tree in trees:
                        if tree[0] != {}:
                            # self.slunet.encode_tree(tree)
                            self.dec.encode_tree(tree)
                else:
                    trees = None
                print(y[0]['entities'])
                ### Commented lines are for beam search decoding
                # slots = self.slunet.inference_beam(y[0]['hidden_states'], self.dec.embed, mmembs, beam=2,
                #                                    yseq=y[0]['yseq'][1:-1], gpthidden=newhidden, nonepenalty=0.0,
                #                                    slothidden=slothidden, slotids=self.sluproc.slotqueries,
                #                                    ptr_gen=y[0]['clspost'] if recog_args.select else y[0]['p_gen'],
                #                                    gptmodel=self.pretrained_LM if self.fullslottext else None,
                #                                    pasttext=(pasttext, self.sluproc.slotquerytext), trees=trees)
                slots = self.slunet.inference(y[0]['hidden_states'], self.dec.embed, mmembs,
                    yseq=y[0]['yseq'][1:-1], gpthidden=newhidden, slothidden=slothidden, slotids=self.sluproc.slotqueries,
                    ptr_gen=y[0]['clspost'] if recog_args.select else y[0]['p_gen'],
                    gptmodel=self.pretrained_LM if self.fullslottext else None,
                    pasttext=(pasttext, self.sluproc.slotquerytext), trees=trees)
                slots = self.sluproc.predict(slots)
                print(slots)
                y[0]['slots'] = slots
                for slot in slots:
                    if slot['filler'] != 'not mentioned':
                        laststate.append(' '.join(slot['type'].split('_') + ['is'] + slot['filler'].split()))
                y[0]['intent_pred'] = '' # intent
                y[0]['shortlist'] = [] # shortlist
                y[0]['yseq_str'] = ''.join([self.char_list[idx] for idx in y[0]['yseq'][1:-1]]).replace('▁', ' ').strip() + '\n'
            best_hid = best_hid if history is not None else None
        return y, best_fusion, best_est, best_hid, ' '.join(laststate)

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        """E2E batch beam search.

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 1. Frontend and encoder
        hs_pad, hlens = self.forward_frontend_and_encoder(xs_pad, ilens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs_pad)
            normalize_score = False
        else:
            lpz = None
            normalize_score = True

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(
            hs_pad,
            hlens,
            lpz,
            recog_args,
            char_list,
            rnnlm,
            normalize_score=normalize_score,
        )

        if prev:
            self.train()
        return y

    def enhance(self, xs):
        """Forward only in the frontend stage.

        :param ndarray xs: input acoustic feature (T, C, F)
        :return: enhaned feature
        :rtype: torch.Tensor
        """
        if self.frontend is None:
            raise RuntimeError("Frontend does't exist")
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, meetings=None, orig_text=None, intents=None,
                                 slots=None, history=None):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            # 1. Frontend and encoder
            hpad, hlens = self.forward_frontend_and_encoder(xs_pad, ilens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)
        self.train()
        return att_ws

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        probs = None
        if self.mtlalpha == 0:
            return probs

        self.eval()
        with torch.no_grad():
            # 1. Frontend and encoder
            hpad, _ = self.forward_frontend_and_encoder(xs_pad, ilens)

            # 2. CTC probs
            probs = self.ctc.softmax(hpad).cpu().numpy()
        self.train()
        return probs

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[:: self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

    def get_mbr_loss(self, ys_pad, batched_hyps, loss_sep=None):
        mbr_loss_list = []
        mbr_loss_unnorm = []
        sum_of_ratios = []
        for i, sample in enumerate(batched_hyps):
            # remove <sos> and <eos>
            if len(sample) > 1:
                y_true = ys_pad[i]
                if loss_sep is not None:
                    normalised_prob = torch.stack([-loss_sep[i]] + [each_hyp['vscore'] for each_hyp in sample])
                else:
                    normalised_prob = torch.stack([each_hyp['vscore'] for each_hyp in sample])
                # seq_lens = torch.Tensor([len(each_hyp['yseq']) for each_hyp in sample])
                normalised_prob = torch.softmax(normalised_prob, dim=-1)

                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_true_text = "".join(seq_true).replace('▁', ' ')
                ref_words = seq_true_text.split()
                werrors = [0] if loss_sep is not None else []
                for j, each_hyp in enumerate(sample):
                    y_hat = each_hyp['yseq'][1:-1]
                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    if self.use_wp_errors:
                        werrors.append(editdistance.eval(seq_hat, seq_true) * self.mwe_factor)
                    else:
                        seq_hat_text = "".join(seq_hat).replace('▁', ' ')
                        hyp_words = seq_hat_text.split()
                        werrors.append(editdistance.eval(hyp_words, ref_words) * self.mwe_factor)
                werrors = torch.tensor(werrors).to(normalised_prob.device)
                mbr_loss = torch.sum(normalised_prob * (werrors - werrors.mean()))
                mbr_loss_list.append(mbr_loss)
                mbr_loss_unnorm.append(torch.sum(normalised_prob * werrors))
        mbr_loss = torch.stack(mbr_loss_list).mean() if len(mbr_loss_list) > 0 else 0
        mbr_loss_unnorm = torch.stack(mbr_loss_unnorm).mean().item() if len(mbr_loss_list) > 0 else 0
        return mbr_loss, mbr_loss_unnorm

    def get_rare_wpe(self, seq_true, KBwplist):
        align_list = []
        word_buffer = []
        count = 1
        for wp in seq_true:
            if wp.endswith('▁'):
                word_buffer.append(wp)
                if tuple(word_buffer) in KBwplist:
                    align_list += [count] * len(word_buffer)
                    count += 1
                else:
                    align_list += [0] * len(word_buffer)
                word_buffer = []
            else:
                word_buffer.append(wp)
        return align_list

    def get_rare_errors(self, wp_align_list, seq_true, step_list):
        cursor = 0
        total_rare_errors = 0
        for error in step_list:
            if cursor < len(wp_align_list) and wp_align_list[cursor] > 0 and error != 'e':
                total_rare_errors += 1
            if error != 'i':
                cursor += 1
        return total_rare_errors

    def get_KBmbr_loss(self, ys_pad, batched_hyps, KBwplist, loss_sep=None):
        mbr_loss_list = []
        mbr_loss_unnorm = []
        KBwordlist = [''.join(word).replace('▁', '') for word in KBwplist]
        for i, sample in enumerate(batched_hyps):
            if len(sample) > 1:
                y_true = ys_pad[i]
                if loss_sep is not None:
                    normalised_prob_GT = torch.stack([-loss_sep[i]] + [each_hyp['vscore'] for each_hyp in sample])
                    normalised_prob_GT = torch.softmax(normalised_prob_GT, dim=-1)
                # else:
                normalised_prob = torch.stack([each_hyp['vscore'] for each_hyp in sample])
                normalised_prob = torch.softmax(normalised_prob, dim=-1)

                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                # get rare word word piece positions
                wp_align_list = self.get_rare_wpe(seq_true, KBwplist)

                seq_true_text = "".join(seq_true).replace('▁', ' ')
                ref_words = seq_true_text.split()
                rare_seq_ref = []
                for word in ref_words:
                    if word in KBwordlist:
                        rare_seq_ref.append(word)
                werrors = [] # [0] if loss_sep is not None else []
                rwerrors = [0] if loss_sep is not None else []
                for j, each_hyp in enumerate(sample):
                    y_hat = each_hyp['yseq'][1:-1]
                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_hat_text = "".join(seq_hat).replace('▁', ' ')
                    hyp_words = seq_hat_text.split()
                    if self.use_wp_errors:
                        werrors.append(editdistance.eval(hyp_words, ref_words))
                        distance = editDistance(seq_true, seq_hat)
                        # werrors.append(distance[len(seq_true)][len(seq_hat)])
                        if rare_seq_ref != []:
                            step_list = getStepList(seq_true, seq_hat, distance)
                            rare_errors = self.get_rare_errors(wp_align_list, seq_true, step_list)
                            rwerrors.append(rare_errors)
                        else:
                            rwerrors.append(0)
                    else:
                        werrors.append(editdistance.eval(hyp_words, ref_words))
                        if rare_seq_ref != []:
                            hyp_KBwords = [word for word in hyp_words if word in KBwordlist]
                            rwerrors.append(editdistance.eval(hyp_KBwords, rare_seq_ref))
                            # werrors[-1] += self.rareweight * rwerrors
                        else:
                            rwerrors.append(0)
                werrors = torch.tensor(werrors).float().to(ys_pad.device)
                rwerrors = torch.tensor(rwerrors).float().to(ys_pad.device)
                # MWE loss
                # mbr_loss = torch.sum(normalised_prob * (werrors - werrors.mean()))
                mwe_loss = torch.sum(normalised_prob * werrors)
                # Rare word error loss
                if loss_sep is not None:
                    mrwe_loss = torch.sum(normalised_prob_GT * rwerrors)
                else:
                    mrwe_loss = torch.sum(normalised_prob * rwerrors)
                # Sum of both losses
                mbr_loss = self.mwe_factor * mwe_loss + self.rareweight * mrwe_loss
                mbr_loss_list.append(mbr_loss)
                mbr_loss_unnorm.append(mbr_loss)
        mbr_loss = torch.stack(mbr_loss_list).mean() if len(mbr_loss_list) > 0 else 0
        mbr_loss_unnorm = torch.stack(mbr_loss_unnorm).mean().item() if len(mbr_loss_list) > 0 else 0
        return mbr_loss, mbr_loss_unnorm
