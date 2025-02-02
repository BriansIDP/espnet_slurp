"""Transducer speech recognition model (pytorch)."""

from argparse import Namespace
from collections import Counter
from dataclasses import asdict
import logging
import math
import numpy
import time

import chainer
import torch
import editdistance
import torch.multiprocessing as mp

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transducer.arguments import (
    add_encoder_general_arguments,  # noqa: H301
    add_rnn_encoder_arguments,  # noqa: H301
    add_custom_encoder_arguments,  # noqa: H301
    add_decoder_general_arguments,  # noqa: H301
    add_rnn_decoder_arguments,  # noqa: H301
    add_custom_decoder_arguments,  # noqa: H301
    add_custom_training_arguments,  # noqa: H301
    add_transducer_arguments,  # noqa: H301
    add_auxiliary_task_arguments,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.auxiliary_task import AuxiliaryTask
from espnet.nets.pytorch_backend.transducer.custom_decoder import CustomDecoder
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoder import DecoderRNNT
from espnet.nets.pytorch_backend.transducer.rnn_encoder import encoder_for
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.transducer.utils import valid_aux_task_layer_list
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.utils.fill_missing_args import fill_missing_args
# gs534 - add KB classes
from espnet.nets.pytorch_backend.KB_utils.KB import KBmeeting, KBmeetingTrain
from espnet.nets.pytorch_backend.KB_utils.KB import KBmeetingTrainContext, Vocabulary
from espnet.nets.pytorch_backend.nets_utils import to_device
# gs534 - MBR loss
from espnet.nets.beam_search_transducer import BeamSearchTransducer


# mp.set_start_method('spawn')

class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer models."""

    def report(self, loss, cer, wer, mbrloss):
        """Instantiate reporter attributes."""
        chainer.reporter.report({"cer": cer}, self)
        chainer.reporter.report({"wer": wer}, self)
        chainer.reporter.report({"loss": loss}, self)
        chainer.reporter.report({"mbrloss": mbrloss}, self)

        logging.info("loss:" + str(loss))


class E2E(ASRInterface, torch.nn.Module):
    """E2E module for transducer models.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (Namespace): argument Namespace containing options
        ignore_id (int): padding symbol id
        blank_id (int): blank symbol id

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments for transducer model."""
        E2E.encoder_add_general_arguments(parser)
        E2E.encoder_add_rnn_arguments(parser)
        E2E.encoder_add_custom_arguments(parser)

        E2E.decoder_add_general_arguments(parser)
        E2E.decoder_add_rnn_arguments(parser)
        E2E.decoder_add_custom_arguments(parser)

        E2E.training_add_custom_arguments(parser)
        E2E.transducer_add_arguments(parser)
        E2E.auxiliary_task_add_arguments(parser)

        return parser

    @staticmethod
    def encoder_add_general_arguments(parser):
        """Add general arguments for encoder."""
        group = parser.add_argument_group("Encoder general arguments")
        group = add_encoder_general_arguments(group)

        return parser

    @staticmethod
    def encoder_add_rnn_arguments(parser):
        """Add arguments for RNN encoder."""
        group = parser.add_argument_group("RNN encoder arguments")
        group = add_rnn_encoder_arguments(group)

        return parser

    @staticmethod
    def encoder_add_custom_arguments(parser):
        """Add arguments for Custom encoder."""
        group = parser.add_argument_group("Custom encoder arguments")
        group = add_custom_encoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_general_arguments(parser):
        """Add general arguments for decoder."""
        group = parser.add_argument_group("Decoder general arguments")
        group = add_decoder_general_arguments(group)

        return parser

    @staticmethod
    def decoder_add_rnn_arguments(parser):
        """Add arguments for RNN decoder."""
        group = parser.add_argument_group("RNN decoder arguments")
        group = add_rnn_decoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_custom_arguments(parser):
        """Add arguments for Custom decoder."""
        group = parser.add_argument_group("Custom decoder arguments")
        group = add_custom_decoder_arguments(group)

        return parser

    @staticmethod
    def training_add_custom_arguments(parser):
        """Add arguments for Custom architecture training."""
        group = parser.add_argument_group("Training arguments for custom archictecture")
        group = add_custom_training_arguments(group)

        return parser

    @staticmethod
    def transducer_add_arguments(parser):
        """Add arguments for transducer model."""
        group = parser.add_argument_group("Transducer model arguments")
        group = add_transducer_arguments(group)

        return parser

    @staticmethod
    def auxiliary_task_add_arguments(parser):
        """Add arguments for auxiliary task."""
        group = parser.add_argument_group("Auxiliary task arguments")
        group = add_auxiliary_task_arguments(group)

        return parser

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        if self.etype == "custom":
            return self.encoder.conv_subsampling_factor * int(
                numpy.prod(self.subsample)
            )
        else:
            return self.enc.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0, training=True):
        """Construct an E2E object for transducer model."""
        torch.nn.Module.__init__(self)

        args = fill_missing_args(args, self.add_arguments)

        self.is_rnnt = True
        self.transducer_weight = args.transducer_weight

        self.use_aux_task = (
            True if (args.aux_task_type is not None and training) else False
        )

        self.use_aux_ctc = args.aux_ctc and training
        self.aux_ctc_weight = args.aux_ctc_weight

        self.use_aux_cross_entropy = args.aux_cross_entropy and training
        self.aux_cross_entropy_weight = args.aux_cross_entropy_weight

        if self.use_aux_task:
            n_layers = (
                (len(args.enc_block_arch) * args.enc_block_repeat - 1)
                if args.enc_block_arch is not None
                else (args.elayers - 1)
            )

            aux_task_layer_list = valid_aux_task_layer_list(
                args.aux_task_layer_list,
                n_layers,
            )
        else:
            aux_task_layer_list = []

        if "custom" in args.etype:
            if args.enc_block_arch is None:
                raise ValueError(
                    "When specifying custom encoder type, --enc-block-arch"
                    "should also be specified in training config. See"
                    "egs/vivos/asr1/conf/transducer/train_*.yaml for more info."
                )

            self.subsample = get_subsample(args, mode="asr", arch="transformer")

            self.encoder = CustomEncoder(
                idim,
                args.enc_block_arch,
                input_layer=args.custom_enc_input_layer,
                repeat_block=args.enc_block_repeat,
                self_attn_type=args.custom_enc_self_attn_type,
                positional_encoding_type=args.custom_enc_positional_encoding_type,
                positionwise_activation_type=args.custom_enc_pw_activation_type,
                conv_mod_activation_type=args.custom_enc_conv_mod_activation_type,
                aux_task_layer_list=aux_task_layer_list,
            )
            encoder_out = self.encoder.enc_out

            self.most_dom_list = args.enc_block_arch[:]
        else:
            self.subsample = get_subsample(args, mode="asr", arch="rnn-t")

            self.enc = encoder_for(
                args,
                idim,
                self.subsample,
                aux_task_layer_list=aux_task_layer_list,
            )
            encoder_out = args.eprojs

        if "custom" in args.dtype:
            if args.dec_block_arch is None:
                raise ValueError(
                    "When specifying custom decoder type, --dec-block-arch"
                    "should also be specified in training config. See"
                    "egs/vivos/asr1/conf/transducer/train_*.yaml for more info."
                )

            self.decoder = CustomDecoder(
                odim,
                args.dec_block_arch,
                input_layer=args.custom_dec_input_layer,
                repeat_block=args.dec_block_repeat,
                positionwise_activation_type=args.custom_dec_pw_activation_type,
                dropout_rate_embed=args.dropout_rate_embed_decoder,
            )
            decoder_out = self.decoder.dunits

            if "custom" in args.etype:
                self.most_dom_list += args.dec_block_arch[:]
            else:
                self.most_dom_list = args.dec_block_arch[:]
        else:
            self.dec = DecoderRNNT(
                odim,
                args.dtype,
                args.dlayers,
                args.dunits,
                blank_id,
                args.dec_embed_dim,
                args.dropout_rate_decoder,
                args.dropout_rate_embed_decoder,
            )
            decoder_out = args.dunits

        # KB related
        bpe = ('<space>' not in args.char_list) # hack here for bpe flag
        self.vocabulary = Vocabulary(args.dictfile, bpe) if args.dictfile != '' else None
        # gs534 - create lexicon tree
        self.meeting_KB = None
        self.n_KBs = getattr(args, 'dynamicKBs', 0)
        self.DBdrop = getattr(args, 'DBdrop', 0.0)
        self.DBinput = getattr(args, 'DBinput', False)
        self.attn_dim = getattr(args, 'attn_dim', args.dunits)
        self.PtrGen = getattr(args, "PtrGen", False)
        self.PtrSche = getattr(args, "PtrSche", False)
        self.init_from = getattr(args, 'init_full_model', None)
        self.smoothprob = getattr(args, 'smoothprob', 1.0)
        self.KBin = getattr(args, 'PtrKBin', 1.0)
        self.treetype = getattr(args, 'treetype', '')
        self.tree_hid = getattr(args, 'treehid', 0)
        if args.meetingKB and args.meetingpath != '':
            if self.n_KBs == 0:
                if args.randomKBsample:
                    self.meeting_KB = KBmeetingTrain(self.vocabulary, args.meetingpath, args.char_list, bpe,
                                                    maxlen=args.KBmaxlen, DBdrop=self.DBdrop)
                else:
                    self.meeting_KB = KBmeeting(self.vocabulary, args.meetingpath, args.char_list, bpe)
            else:
                self.meeting_KB = KBmeetingTrainContext(self.vocabulary, args.meetingpath, args.char_list, bpe,
                                                    maxlen=args.KBmaxlen, DBdrop=self.DBdrop)
            if self.DBinput:
                self.DBembed = torch.nn.Linear(odim, self.attn_dim, bias=False)
            self.dropout_KB = torch.nn.Dropout(p=args.dropout_rate_embed_decoder)
            # Use pointer generator
            if self.PtrGen:
                self.embdim = args.dec_embed_dim
                # Define graph neural net parameters
                self.tree_hid = self.embdim if self.tree_hid == 0 else self.tree_hid
                if self.treetype.startswith('gcn'):
                    self.gcn_l1 = torch.nn.Linear(self.embdim, self.tree_hid)
                    self.gcn_l2 = torch.nn.Linear(self.tree_hid, self.tree_hid)
                    if self.treetype[-1] == '3':
                        self.gcn_l3 = torch.nn.Linear(self.tree_hid, self.tree_hid)
                else:
                    self.recursive_proj = torch.nn.Linear(self.embdim+self.tree_hid, self.tree_hid)

                self.ooKBemb = torch.nn.Embedding(1, self.tree_hid)
                self.Qproj_char = torch.nn.Linear(args.dec_embed_dim, self.attn_dim)
                # self.Qproj_char = torch.nn.Linear(decoder_out, self.attn_dim)
                self.Qproj_acoustic = torch.nn.Linear(encoder_out, self.attn_dim)
                self.Kproj = torch.nn.Linear(self.tree_hid, self.attn_dim)
                self.pointer_gate = torch.nn.Linear(self.attn_dim+args.joint_dim, 1)

        self.joint_network = JointNetwork(
            odim, encoder_out, decoder_out, args.joint_dim, args.joint_activation_type,
            useKB=(args.meetingKB and (self.DBinput or self.KBin)), plm_dim=self.attn_dim
        )

        if hasattr(self, "most_dom_list"):
            self.most_dom_dim = sorted(
                Counter(
                    d["d_hidden"] for d in self.most_dom_list if "d_hidden" in d
                ).most_common(),
                key=lambda x: x[0],
                reverse=True,
            )[0][0]

        self.etype = args.etype
        self.dtype = args.dtype

        self.char_list = args.char_list
        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim

        self.reporter = Reporter()
        self.error_calculator = None

        self.default_parameters(args)

        # gs534 - MBR training
        self.use_mbrloss = getattr(args, 'mbrloss', False)
        self.mbrbeam = getattr(args, 'mbrbeam', 1)
        self.mbrnbest = getattr(args, 'mbrnbest', 1)
        self.mbrlambda = getattr(args, 'mbrlambda', 1.0)
        self.rareweight = getattr(args, 'mbrrareweight', 0.0)
        self.mwe_factor = getattr(args, 'mweweight', 1.0)
        self.use_wp_errors = getattr(args, 'use_wp_errors', False)
        self.cfm_mbr_start = getattr(args, 'cfm_mbr_start', 0)
        self.wrt_alignments = getattr(args, 'wrt_alignments', False)

        if training:
            self.criterion = TransLoss(args.trans_type, self.blank_id, useKB=args.meetingKB)

            decoder = self.decoder if self.dtype == "custom" else self.dec

            if args.report_cer or args.report_wer:
                self.error_calculator = ErrorCalculator(
                    decoder,
                    self.joint_network,
                    args.char_list,
                    args.sym_space,
                    args.sym_blank,
                    args.report_cer,
                    args.report_wer,
                )

            if self.use_aux_task:
                self.auxiliary_task = AuxiliaryTask(
                    decoder,
                    self.joint_network,
                    self.criterion,
                    args.aux_task_type,
                    args.aux_task_weight,
                    encoder_out,
                    args.joint_dim,
                )

            if self.use_aux_ctc:
                self.aux_ctc = ctc_for(
                    Namespace(
                        num_encs=1,
                        eprojs=encoder_out,
                        dropout_rate=args.aux_ctc_dropout_rate,
                        ctc_type="warpctc",
                    ),
                    odim,
                )

            if self.use_aux_cross_entropy:
                self.aux_decoder_output = torch.nn.Linear(decoder_out, odim)

                self.aux_cross_entropy = LabelSmoothingLoss(
                    odim, ignore_id, args.aux_cross_entropy_smoothing
                )
            # gs534 - MBR training
            if self.use_mbrloss:
                KBmodules = {'meeting_KB': None, 'DBinput': False, 'PtrGen': False}
                if self.meeting_KB is not None:
                    KBmodules['meeting_KB'] = self.meeting_KB
                    KBmodules['char_list'] = self.char_list
                    if self.DBinput:
                        KBmodules['DBembed'] = self.DBembed
                        KBmodules['DBinput'] = True
                    if self.PtrGen:
                        KBmodules['PtrGen'] = True
                        KBmodules['ooKBemb'] = self.ooKBemb
                        KBmodules['Qproj_char'] = self.Qproj_char
                        KBmodules['Qproj_acoustic'] = self.Qproj_acoustic
                        KBmodules['Kproj'] = self.Kproj
                        KBmodules['pointer_gate'] = self.pointer_gate
                        KBmodules['smoothprob'] = getattr(self, 'smoothprob', 1.0)
                        KBmodules['KBin'] = getattr(self, 'KBin', False)
                        KBmodules['dropout_KB'] = getattr(self, 'dropout_KB', None)

                self.beam_search = BeamSearchTransducer(
                    decoder=self.dec,
                    joint_network=self.joint_network,
                    search_type="batched",
                    beam_size=self.mbrbeam,
                    nbest=self.mbrnbest,
                    score_norm=False,
                    KBmodules=KBmodules,
                    char_list=self.char_list
                )

        self.loss = None
        self.rnnlm = None

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer.

        Args:
            args (Namespace): argument Namespace containing options

        """
        initializer(self, args)
        # gs534 - lextree
        # if self.init_from is not None:
        #     model_init = torch.load(self.init_from, map_location=lambda storage, loc: storage)
        #     model_init = model_init.state_dict() if not isinstance(model_init, dict) else model_init
        #     own_state = self.state_dict()
        #     for name, param in model_init.items():
        #         if name in own_state:
        #             own_state[name].copy_(param.data)

    def synchronise_params(self):
        self.beam_search.decoder.set_device(torch.device('cpu'))
        self.beam_search.decoder.set_data_type(self.dec.data_type)
        decoder_state = self.dec.state_dict()
        joint_net_state = self.joint_network.state_dict()
        bs_dec_state = self.beam_search.decoder.state_dict()
        bs_joint_state = self.beam_search.joint_network.state_dict()
        for name, param in decoder_state.items():
            if name in bs_dec_state:
                bs_dec_state[name].copy_(param.data)
        for name, param in joint_net_state.items():
            if name in bs_joint_state:
                bs_joint_state[name].copy_(param.data)

    def forward_encoder(self, xs_pad, ilens):
        xs_pad = xs_pad[:, : max(ilens)]
        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None
        return hs_pad, aux_hs_pad, hs_mask

    def forward(self, xs_pad, ilens, ys_pad, meetings=None, orig_text=None, intents=None, slots=None):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value

        """
        # 1. encoder
        hs_pad, aux_hs_pad, hs_mask = self.forward_encoder(xs_pad, ilens)

        # 1.5. transducer preparation related
        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_mask
        )

        # 2. decoder
        if "custom" in self.dtype:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            pred_pad = self.dec(hs_pad, ys_in_pad)

        # Get KB information
        hs_plm = None
        ptr_dist = None
        p_gen = None
        meeting_info = None
        if self.meeting_KB is not None and self.dec.epoch >= self.PtrSche:
            # self.meeting_KB.DBdrop = self.DBdrop if self.training else 0
            meeting_info = self.meeting_KB.get_meeting_KB(meetings, ilens.size(0))
            # get padded text sequence
            ys_ptrgen_pad = torch.cat([ys_in_pad[:, 0:1], ys_pad], dim=-1)

            # get tree net encodings
            self.encode_tree(meeting_info[2][0])
            trees = meeting_info[2]
            p_gen_mask_all = []

            if self.DBinput:
                hs_plm = self.dropout_KB(self.DBembed((1-lex_masks[:, :, :-1].float())).unsqueeze(1))

            if self.PtrGen:
                query_acoustic = self.Qproj_acoustic(hs_pad) # nutts * T * attn_dim
                KBembedding = []
                ptr_dist = []
                for i in range(ys_in_pad.size(1)):
                    step_mask, trees, p_gen_mask, step_embs, back_transform = self.get_lextree_step_embs(
                        ys_ptrgen_pad[:, i], trees, meeting_info[2])
                    p_gen_mask_all.append(p_gen_mask)
                    query_char = self.dec.dropout_embed(self.dec.embed(ys_in_pad[:,i])) # nutts * embdim
                    query_char = self.Qproj_char(query_char).unsqueeze(1) # nutts * 1 * attn_dim
                    query = query_char + query_acoustic # nutts * T * attn_dim
                    KBembedding_i, ptr_dist_i = self.get_meetingKB_emb_map(query, step_mask, step_embs, back_transform)
                    KBembedding.append(KBembedding_i.unsqueeze(2)) # nutts * T * attn_dim
                    ptr_dist.append(ptr_dist_i.unsqueeze(2)) # nutts * T * nbpe
                KBembedding = torch.cat(KBembedding, dim=2)
                ptr_dist = torch.cat(ptr_dist, dim=2)
                if self.KBin and not self.DBinput:
                    hs_plm = self.dropout_KB(KBembedding)

        # Forward joint network
        z, joint_activations = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1), h_plm=hs_plm)

        # Process pointer generator
        if self.meeting_KB is not None and self.PtrGen and self.dec.epoch >= self.PtrSche:
            p_gen = torch.sigmoid(self.pointer_gate(torch.cat((joint_activations, KBembedding), dim=-1)))
            ptr_mask = to_device(self, torch.tensor(p_gen_mask_all)).t()
            p_gen = p_gen.masked_fill(ptr_mask.unsqueeze(1).unsqueeze(-1).bool(), 0) * self.smoothprob

        # 3. loss computation
        loss_trans = self.criterion(z, target, pred_len, target_len, ptr_dist=ptr_dist, p_gen=p_gen)

        # gs534 - MBR training
        all_hyps = []
        self.mbrloss = 0
        mbrloss_unnorm = 0
        if self.use_mbrloss and self.dec.epoch >= self.cfm_mbr_start:
            if self.wrt_alignments:
                # self.synchronise_params()
                with torch.no_grad():
                    # inputs = zip([hs.data.cpu() for hs in hs_pad], [ys.data.cpu() for ys in ys_pad])
                    # results = self.pool.starmap(self.beam_search.second_search_algorithm, inputs)
                    all_hyps = self.beam_search.search_algorithm(hs_pad, hs_mask, meeting_info)
                    results = []
                    for k, hyps in enumerate(all_hyps):
                        seq_true = [self.char_list[int(idx)] for idx in ys_pad[k] if int(idx) != -1]
                        seq_true_text = "".join(seq_true).replace('▁', ' ')
                        ref_words = seq_true_text.split()
                        result = []
                        for hyp in hyps:
                            seq_hat = [self.char_list[int(idx)] for idx in hyp.yseq if int(idx) != 0]
                            seq_hat_text = seq_hat_text = "".join(seq_hat).replace('▁', ' ')
                            hyp_words = seq_hat_text.split()
                            werror = editdistance.eval(hyp_words, ref_words)
                            result.append((hyp.yseq[1:], hyp.score, werror))
                        results.append(result)
                self.mbrloss = self.get_KBmbr_loss_alignments(results, hs_pad, hs_mask)
                mbrloss_unnorm = self.mbrloss.item()
            else:
                all_hyps = self.beam_search.search_algorithm(hs_pad, hs_mask, meeting_info)
                KBwplist = meeting_info[1] if self.meeting_KB is not None else []
                self.mbrloss, mbrloss_unnorm = self.get_KBmbr_loss(ys_pad, all_hyps, KBwplist)

        if self.use_aux_task and aux_hs_pad is not None:
            loss_trans += self.auxiliary_task(
                aux_hs_pad, pred_pad, z, target, pred_len, target_len
            )

        if self.use_aux_ctc:
            if "custom" in self.etype:
                hs_mask = torch.IntTensor(
                    [h.size(1) for h in hs_mask],
                ).to(hs_mask.device)

            loss_ctc = self.aux_ctc(hs_pad, hs_mask, ys_pad)
        else:
            loss_ctc = 0

        if self.use_aux_cross_entropy:
            loss_ce = self.aux_cross_entropy(
                self.aux_decoder_output(pred_pad), ys_out_pad
            )
        else:
            loss_ce = 0

        loss = (
            self.transducer_weight * loss_trans
            + self.aux_ctc_weight * loss_ctc
            + self.aux_cross_entropy_weight * loss_ce
        )
        if self.use_mbrloss:
            loss = loss * self.mbrlambda + self.mbrloss

        self.loss = loss
        loss_data = float(loss)

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad, ys_pad)

        if not math.isnan(loss_data):
            self.reporter.report(loss_data, cer, wer, mbrloss_unnorm)
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def encode_custom(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, D_enc)

        """
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)

        return enc_output.squeeze(0)

    def encode_rnn(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, D_enc)

        """
        p = next(self.parameters())

        ilens = [x.shape[0]]
        x = x[:: self.subsample[0], :]

        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        hs = h.contiguous().unsqueeze(0)

        hs, _, _ = self.enc(hs, ilens)

        return hs.squeeze(0)

    def recognize(self, x, beam_search, meetings=None, estlm=None, estlm_factor=0.0,
                  prev_hid=None):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            beam_search (class): beam search class

        Returns:
            nbest_hyps (list): n-best decoding results

        """
        self.eval()

        if "custom" in self.etype:
            h = self.encode_custom(x)
        else:
            h = self.encode_rnn(x)

        nbest_hyps = beam_search(h, meetings=meetings, estlm=estlm, estlm_factor=estlm_factor, 
                                 prev_hid=None)

        return [asdict(n) for n in nbest_hyps]

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, meetings=None):
        """E2E attention calculation.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax)

        Returns:
            ret (ndarray): attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).

        """
        self.eval()

        if "custom" not in self.etype and "custom" not in self.dtype:
            return []
        else:
            with torch.no_grad():
                self.forward(xs_pad, ilens, ys_pad)

            ret = dict()
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) or isinstance(
                    m, RelPositionMultiHeadedAttention
                ):
                    ret[name] = m.attn.cpu().numpy()

        self.train()

        return ret

    def get_lextree_encs(self, lextree, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            ey = self.dec.embed(to_device(self, torch.LongTensor([wordpiece])))
            ey = self.dropout_KB(ey)
            lextree.append(ey)
            return ey
        elif lextree[1] == -1:
            wordpieces = []
            for newpiece, values in lextree[0].items():
                wordpieces.append(self.get_lextree_encs(values, newpiece))
            wordpiece_h = torch.cat(wordpieces, dim=0).sum(dim=0).unsqueeze(0)
            if wordpiece is not None:
                ey = self.dec.embed(to_device(self, torch.LongTensor([wordpiece])))
                ey = self.dropout_KB(ey)
                wordpiece_h = self.recursive_proj(torch.cat([ey, wordpiece_h], dim=-1))
                wordpiece_h = torch.relu(wordpiece_h)
                # wordpiece_h = torch.sigmoid(wordpiece_h)
                lextree.append(wordpiece_h)
            return wordpiece_h

    def get_lextree_encs_gcn(self, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = len(embeddings)
            # ey = self.dec.embed(to_device(self, torch.LongTensor([wordpiece])))
            ey = self.dec.embed.weight[wordpiece].unsqueeze(0)
            embeddings.append(self.dropout_KB(ey))
            adjacency.append([idx])
            lextree.append([])
            lextree.append(idx)
            return idx
        elif lextree[1] == -1 and lextree[0] != {}:
            ids = []
            idx = len(embeddings)
            if wordpiece is not None:
                # ey = self.dec.embed(to_device(self, torch.LongTensor([wordpiece])))
                ey = self.dec.embed.weight[wordpiece].unsqueeze(0)
                embeddings.append(self.dropout_KB(ey))
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
        if self.treetype != 'gcn' and int(self.treetype[-1]) > 1:
            nodes_encs = self.gcn_l2(nodes_encs)
            nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, nodes_encs))
            if int(self.treetype[-1]) > 2:
                nodes_encs = self.gcn_l3(nodes_encs)
                nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, nodes_encs))
        return nodes_encs

    def fill_lextree_encs_gcn(self, lextree, nodes_encs, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = lextree[4]
            # lextree.append(nodes_encs[idx])
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[1] == -1 and lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs_gcn(values, nodes_encs, newpiece)
            # lextree.append(nodes_encs[idx])
            lextree[3] = nodes_encs[idx].unsqueeze(0)

    def encode_tree(self, prefixtree):
        if self.treetype.startswith('gcn'):
            embeddings, adjacency = [], []
            self.get_lextree_encs_gcn(prefixtree, embeddings, adjacency)
            nodes_encs = self.forward_gcn(prefixtree, embeddings, adjacency)
            self.fill_lextree_encs_gcn(prefixtree, nodes_encs)
        else:
            self.get_lextree_encs(prefixtree)

    def get_meetingKB_emb_map(self, query, meeting_mask, meeting_embs, back_transform):
        """
        query: (nutts, T, attn_dim)
        meeting_embs: (nutts, n_nodes, embdim)
        meeting_mask: (nutts, n_nodes)
        back_transform: (nutts, n_nodes, nbpes)

        return: (nutts, T, nbpes)
        """
        meeting_KB = self.dropout_KB(self.Kproj(meeting_embs))
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

    def get_lextree_step_embs(self, char_ids, trees, origTries):
        ooKB_id = len(self.char_list)
        maxlen = 0
        index_list = []
        new_trees = []
        p_gen_mask = []
        step_embs = []
        for i, vy in enumerate(char_ids):
            new_tree = trees[i][0]
            vy = vy.item()
            if vy in [self.blank_id] or self.char_list[vy].endswith('▁'):
                new_tree = origTries[i]
                p_gen_mask.append(0)
            elif vy not in new_tree:
                new_tree = [{}]
                p_gen_mask.append(1)
            else:
                new_tree = new_tree[vy]
                p_gen_mask.append(0)
            new_trees.append(new_tree)
            if len(new_tree[0].keys()) > maxlen:
                maxlen = len(new_tree[0].keys())
            index_list.append(list(new_tree[0].keys()))
            if len(new_tree) > 2:
                step_embs.append(torch.cat([node[3] for key, node in new_tree[0].items()], dim=0))
            else:
                step_embs.append(to_device(self, torch.empty(0, self.tree_hid)))
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
        return step_mask, new_trees, p_gen_mask, step_embs, back_transform

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
                    normalised_prob_GT = torch.stack([-loss_sep[i]] + [each_hyp.vscore for each_hyp in sample])
                    normalised_prob_GT = torch.softmax(normalised_prob_GT, dim=-1)
                # else:
                normalised_prob = torch.cat([each_hyp.vscore for each_hyp in sample])
                normalised_prob = torch.softmax(normalised_prob, dim=-1)

                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                # get rare word word piece positions
                if self.use_wp_errors:
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
                    y_hat = each_hyp.yseq[1:]
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

    def get_KBmbr_loss_alignments(self, results, hs_pad, hs_mask):
        # yseqs, wers, logprobs = zip(*results)
        maxyseqlen = 0
        scale_factors = []
        risks = []
        for sample in results:
            for hyp in sample:
                yseq = hyp[0]
                if len(yseq) > maxyseqlen:
                    maxyseqlen = len(yseq)
        new_yseqs_pad = []
        new_hs_mask = []
        new_hs_pad = []
        for i, sample in enumerate(results):
            hyp_logprobs = []
            werrors = []
            for hyp in sample:
                yseq = hyp[0]
                hyp_logprobs.append(hyp[1])
                werrors.append(float(hyp[2]))
                new_yseqs_pad.append(yseq + [-1] * (maxyseqlen - len(yseq)))
                new_hs_mask.append(hs_mask[i])
                new_hs_pad.append(hs_pad[i])
            hyp_logprobs = torch.softmax(torch.tensor(hyp_logprobs), dim=-1) # P_hat
            werrors = torch.tensor(werrors)
            factors = hyp_logprobs * (werrors - (hyp_logprobs * werrors).sum())
            scale_factors.append(factors)
        new_yseqs_pad = torch.LongTensor(new_yseqs_pad)
        new_hs_mask = torch.stack(new_hs_mask)
        new_hs_pad = torch.stack(new_hs_pad)
        scale_factors = torch.cat(scale_factors, dim=0).to(new_hs_pad.device)
        # Do forward pass again
        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            new_yseqs_pad, new_hs_mask
        )
        pred_pad = self.dec(new_hs_pad, ys_in_pad.to(new_hs_pad.device))
        hs_plm = None
        p_gen = None
        ptr_dist = None
        z, joint_activations = self.joint_network(new_hs_pad.unsqueeze(2), pred_pad.unsqueeze(1), h_plm=hs_plm)
        loss_trans = self.criterion(z, target.to(z.device), pred_len.to(z.device), target_len.to(z.device),
            ptr_dist=ptr_dist, p_gen=p_gen, reduction='none')
        loss_trans = (-loss_trans * scale_factors).sum() / len(results)
        return loss_trans
