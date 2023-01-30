# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conformer speech recognition model (pytorch).

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

import torch
from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.rnn.argument import add_arguments_conformer_encoder_common
from espnet.nets.pytorch_backend.e2e_asr import E2E as E2ELas
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class E2E(E2ELas):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.add_conformer_encoder_arguments(parser)
        E2ELas.attention_add_arguments(parser)
        E2ELas.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def add_conformer_encoder_arguments(parser):
        """Add arguments for conformer model."""
        group = parser.add_argument_group("conformer model specific setting")
        group = add_arguments_conformer_encoder_common(group)
        return parser

    def __init__(self, idim, odim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(idim, odim, args)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        if args.transformer_adim is None:
            args.transformer_adim = args.adim
        if args.transformer_aheads is None:
            args.transformer_aheads = args.aheads
        self.enc = Encoder(
            idim=idim,
            attention_dim=args.transformer_adim,
            attention_heads=args.transformer_aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            input_layer=args.transformer_input_layer,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
        )
        # gs534 - lextree
        if self.init_from is not None:
            self.load_from()

    def forward_frontend_and_encoder(self, xs_pad, ilens):
        """Forward front-end and encoder, different from LAS."""
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.enc(xs_pad, src_mask)
        hlens = torch.sum(hs_mask.squeeze(-2), dim=-1)
        return hs_pad, hlens
