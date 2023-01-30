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
from espnet.nets.pytorch_backend.KB_utils.wer import editDistance, getStepList
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss, mbrloss):
        """Report at every step."""
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"loss_MBR": mbrloss}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer_ctc": cer_ctc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
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
        self.mbr_use_KB = getattr(args, 'mbruseKB', False) if getattr(args, 'mbrloss', False) else True
        if getattr(args, 'meetingKB', False) and getattr(args, 'meetingpath', '') != '':
            self.KBminlen = getattr(args, 'KBminlen', args.KBmaxlen)
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
            meetingKB=self.meeting_KB if self.mbr_use_KB else None)

        # weight initialization
        self.init_from = getattr(args, 'init_full_model', None)
        self.init_like_chainer()

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
        # gs534 - lextree
        # if self.init_from is not None:
        #     self.load_from()

    def load_from(self):
        model_init = torch.load(self.init_from, map_location=lambda storage, loc: storage)
        model_init = model_init.state_dict() if not isinstance(model_init, dict) else model_init
        # import pdb; pdb.set_trace()
        self.load_state_dict(model_init, strict=False)
        # own_state = self.state_dict()
        # for name, param in model_init.items():
        #     if name in own_state:
        #         own_state[name].copy_(param.data)
    
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

    def forward(self, xs_pad, ilens, ys_pad, meetings=None):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
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
            if self.meeting_KB is not None:
                # self.meeting_KB.DBdrop = self.DBdrop if self.training else 0
                meeting_info = self.meeting_KB.get_meeting_KB(meetings, ilens.size(0))

            self.loss_att, acc, _, lossGT = self.dec(hs_pad, hlens, ys_pad,
                meeting_info=meeting_info if self.mbr_use_KB else None, useGT=self.useGT)

            # gs534 - MBR training
            if self.use_mbrloss and self.dec.epoch >= self.cfm_mbr_start:
                #  set maximum decoding length to avoid running out of mem
                y_maxlen = min(int(ys_pad.size(1) * 1.5), hlens.max(), 210)
                # print(ys_pad.size())
                nbest = self.dec.batch_decode_nbest(hs_pad, hlens, beam=self.mbrbeam,
                                                    nbest=self.mbrnbest, maxlen=y_maxlen, meeting_info=meeting_info)
                lossGT = lossGT.view(ys_pad.size(0), -1).sum(1) if self.useGT else None
                if self.meeting_KB is not None:
                    self.mbrloss, self.mbrloss_unnorm = self.get_KBmbr_loss_new(ys_pad, nbest, meeting_info[1], loss_sep=lossGT)
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
            self.loss = self.loss_att
            loss_att_data = float(self.loss_att)
            loss_mbr_data = float(self.mbrloss_unnorm)
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
                loss_ctc_data, loss_att_data, acc, cer_ctc, cer, wer, loss_data, loss_mbr_data
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
                  estlm=None, estlm_factor=0.0, sel_lm=None, prev_hid=None):
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
        if self.meeting_KB is not None and not recog_args.select:
            meeting_info = self.meeting_KB.get_meeting_KB(meetings, 1)
        elif self.meeting_KB is not None and recog_args.select:
            meeting_info = (self.meeting_KB.full_wordlist, meetings if recog_args.nbestKB else None,
                self.meeting_KB.full_lextree)

        # 2. Decoder
        # decode the first utterance
        y, best_hid = self.dec.recognize_beam(hs[0], lpz, recog_args, char_list, rnnlm,
                                    meeting_info=meeting_info, ranking_norm=recog_args.ranking_norm,
                                    estlm=estlm, estlm_factor=estlm_factor, sel_lm=sel_lm,
                                    topk=recog_args.topk, prev_hid=prev_hid,classlm=recog_args.classlm)
        return y, best_hid

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

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, meetings=None):
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
                # word_ref_lens.append(len(ref_words))
                # ref_chars = seq_true_text.replace(' ', '')
                # char_ref_lens.append(len(ref_chars))
                # werrors = ys_pad.new_zeros(len(sample), dtype=torch.float)
                werrors = [0] if loss_sep is not None else []
                for j, each_hyp in enumerate(sample):
                    y_hat = each_hyp['yseq'][1:-1]
                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_hat_text = "".join(seq_hat).replace('▁', ' ')
                    hyp_words = seq_hat_text.split()
                    werrors.append(editdistance.eval(hyp_words, ref_words) * self.mwe_factor)
                    # hyp_chars = seq_hat_text.replace(' ', '')
                    # char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                werrors = torch.tensor(werrors).to(normalised_prob.device)
                mbr_loss = torch.sum(normalised_prob * (werrors - werrors.mean()))
                mbr_loss_list.append(mbr_loss)
                mbr_loss_unnorm.append(torch.sum(normalised_prob * werrors))
        mbr_loss = torch.stack(mbr_loss_list).mean() if len(mbr_loss_list) > 0 else 0 
        mbr_loss_unnorm = torch.stack(mbr_loss_unnorm).mean().item() if len(mbr_loss_list) > 0 else 0
        return mbr_loss, mbr_loss_unnorm

    def get_KBmbr_loss(self, ys_pad, batched_hyps, KBwordlist, loss_sep=None):
        mbr_loss_list = []
        mbr_loss_unnorm = []
        KBwordlist = [''.join(word).replace('▁', '') for word in KBwordlist]
        for i, sample in enumerate(batched_hyps):
            if len(sample) > 1:
                y_true = ys_pad[i]
                if loss_sep is not None:
                    normalised_prob = torch.stack([-loss_sep[i]] + [each_hyp['vscore'] for each_hyp in sample])
                else:
                    normalised_prob = torch.stack([each_hyp['vscore'] for each_hyp in sample])
                normalised_prob = torch.softmax(normalised_prob, dim=-1)

                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_true_text = "".join(seq_true).replace('▁', ' ')
                ref_words = seq_true_text.split()
                rare_seq_ref = []
                for word in ref_words:
                    if word in KBwordlist:
                        rare_seq_ref.append(word)
                werrors = [0] if loss_sep is not None else []
                for j, each_hyp in enumerate(sample):
                    y_hat = each_hyp['yseq'][1:-1]
                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_hat_text = "".join(seq_hat).replace('▁', ' ')
                    hyp_words = seq_hat_text.split()
                    werrors.append(editdistance.eval(hyp_words, ref_words) * self.mwe_factor)
                    if rare_seq_ref != []:
                        hyp_KBwords = [word for word in hyp_words if word in KBwordlist]
                        rwerrors = editdistance.eval(hyp_KBwords, rare_seq_ref)
                        werrors[-1] += self.rareweight * rwerrors
                werrors = torch.tensor(werrors).to(ys_pad.device)
                mbr_loss = torch.sum(normalised_prob * (werrors - werrors.mean()))
                # mbr_loss = torch.sum(normalised_prob * werrors)
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

    def get_KBmbr_loss_new(self, ys_pad, batched_hyps, KBwplist, loss_sep=None):
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
                    werrors.append(editdistance.eval(hyp_words, ref_words))
                    if self.use_wp_errors:
                        # werrors.append(editdistance.eval(seq_hat, seq_true))
                        # werrors.append(distance[len(seq_true)][len(seq_hat)])
                        if rare_seq_ref != []:
                            distance = editDistance(seq_true, seq_hat)
                            step_list = getStepList(seq_true, seq_hat, distance)
                            rare_errors = self.get_rare_errors(wp_align_list, seq_true, step_list)
                            rwerrors.append(rare_errors if rare_errors != 0 else 0)
                        else:
                            rwerrors.append(0)
                    else:
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
