# Copyright 2020 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer common arguments."""

from espnet.utils.cli_utils import strtobool
from espnet.nets.pytorch_backend.conformer.argument import add_arguments_conformer_common


def add_arguments_rnn_encoder_common(group):
    """Define common arguments for RNN encoder."""
    group.add_argument(
        "--etype",
        default="blstmp",
        type=str,
        choices=[
            "lstm",
            "blstm",
            "lstmp",
            "blstmp",
            "vgglstmp",
            "vggblstmp",
            "vgglstm",
            "vggblstm",
            "cnnlstmp",
            "cnnblstmp",
            "cnnlstm",
            "cnnblstm",
            "gru",
            "bgru",
            "grup",
            "bgrup",
            "vgggrup",
            "vggbgrup",
            "vgggru",
            "vggbgru",
            "cnngrup",
            "cnnbgrup",
            "cnngru",
            "cnnbgru",
        ],
        help="Type of encoder network architecture",
    )
    group.add_argument(
        "--elayers",
        default=4,
        type=int,
        help="Number of encoder layers",
    )
    group.add_argument(
        "--eunits",
        "-u",
        default=300,
        type=int,
        help="Number of encoder hidden units",
    )
    group.add_argument(
        "--eprojs", default=320, type=int, help="Number of encoder projection units"
    )
    group.add_argument(
        "--subsample",
        default="1",
        type=str,
        help="Subsample input frames x_y_z means "
        "subsample every x frame at 1st layer, "
        "every y frame at 2nd layer etc.",
    )
    group.add_argument(
        "--apply-bn",
        default=False,
        type=strtobool,
        help="Apply batch normalization after each projection layer for RNNP"
    )
    # Noam relaated
    # group.add_argument(
    #     "--transformer-lr",
    #     default=10.0,
    #     type=float,
    #     help="Initial value of learning rate",
    # )
    # group.add_argument(
    #     "--transformer-warmup-steps",
    #     default=25000,
    #     type=int,
    #     help="optimizer warmup steps",
    # )
    return group


def add_arguments_conformer_encoder_common(group):
    """Define common arguments for conformer encoder."""
    group = add_arguments_rnn_encoder_common(group)
    group = add_arguments_conformer_common(group)
    group.add_argument(
        "--transformer-adim",
        default=320,
        type=int,
        help="Number of attention transformation dimensions",
    )
    group.add_argument(
        "--transformer-aheads",
        default=4,
        type=int,
        help="Number of heads for multi head attention",
    )
    group.add_argument(
        "--transformer-attn-dropout-rate",
        default=None,
        type=float,
        help="dropout in transformer attention. use --dropout-rate if None is set",
    )
    group.add_argument(
        "--transformer-input-layer",
        type=str,
        default="conv2d",
        choices=["conv2d", "linear", "embed"],
        help="transformer input layer type",
    )
    group.add_argument(
        "--transformer-encoder-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "rel_selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer encoder self-attention layer type",
    )
    group.add_argument(
        "--transformer-lr",
        default=10.0,
        type=float,
        help="Initial value of learning rate",
    )
    group.add_argument(
        "--transformer-warmup-steps",
        default=25000,
        type=int,
        help="optimizer warmup steps",
    )
    return group

def add_arguments_rnn_decoder_common(group):
    """Define common arguments for RNN decoder."""
    group.add_argument(
        "--dtype",
        default="lstm",
        type=str,
        choices=["lstm", "gru"],
        help="Type of decoder network architecture",
    )
    group.add_argument(
        "--dlayers", default=1, type=int, help="Number of decoder layers"
    )
    group.add_argument(
        "--dunits", default=320, type=int, help="Number of decoder hidden units"
    )
    group.add_argument(
        "--dropout-rate-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder",
    )
    group.add_argument(
        "--sampling-probability",
        default=0.0,
        type=float,
        help="Ratio of predicted labels fed back to decoder",
    )
    group.add_argument(
        "--lsm-type",
        const="",
        default="",
        type=str,
        nargs="?",
        choices=["", "unigram"],
        help="Apply label smoothing with a specified distribution type",
    )
    group.add_argument(
        "--context-residual",
        default=False,
        type=strtobool,
        nargs="?",
        help="The flag to switch to use context vector residual in the decoder network",
    )
    # group.add_argument('--meetingpath', type=str, default='',
    #                    help='The path to meeting level KB')
    # group.add_argument('--dictfile', type=str, default='',
    #                    help='Dictionary for word level LM')
    # group.add_argument('--lm-odim', type=int, default=0,
    #                    help='Language model output dimension')
    # group.add_argument('--attn-label', type=strtobool, default=False,
    #                    help='Load attention label for KB')
    # group.add_argument('--wordemb', default=0, type=int,
    #                    help='Number of word embedding hidden units')
    # group.add_argument('--PtrGen', default=False, type=strtobool,
    #                        help='use pointer generator for KB')
    # group.add_argument('--KBlextree', default=False, type=strtobool,
    #                    help='use lexicon tree to represent KB')
    # group.add_argument('--dynamicKBs', default=0, type=int,
    #                    help='Number of different KB arrangements')
    # group.add_argument('--PtrSche', default=0, type=int,
    #                    help='Pointer net scheduler')
    # group.add_argument('--PtrKBin', default=False, type=strtobool,
    #                    help='Use KB embedding as auxiliary input when PtrGen')
    # group.add_argument('--smoothprob', default=1.0, type=float,
    #                    help='KB probability smoothing factor')
    # group.add_argument('--attn_dim', default=1024, type=int,
    #                    help='KB lookup attention space')
    # group.add_argument('--KBlossfactor', default=0.0, type=float,
    #                    help='Factor of KB loss')
    # group.add_argument('--acousticonly', default=False, type=strtobool,
    #                    help='KB lookup only use acoustic information')
    # group.add_argument('--additive-attn', default=False, type=strtobool,
    #                    help='KB lookup uses additive attention')
    # group.add_argument('--ooKBemb', default=False, type=strtobool,
    #                    help='out-of-KB embedding for word piece')
    # group.add_argument('--KBmaxlen', default=500, type=int,
    #                    help='KB max sizes')
    # group.add_argument('--DBmask', default=0.0, type=float,
    #                    help='Probability of masking from deep biasing')
    # group.add_argument('--DBdrop', default=0.0, type=float,
    #                    help='Probability of dropping bias entries')
    # group.add_argument('--DBinput', default=False, type=strtobool,
    #                    help='Use Deep biasing input')
    return group


def add_arguments_rnn_attention_common(group):
    """Define common arguments for RNN attention."""
    group.add_argument(
        "--atype",
        default="dot",
        type=str,
        choices=[
            "noatt",
            "dot",
            "add",
            "location",
            "coverage",
            "coverage_location",
            "location2d",
            "location_recurrent",
            "multi_head_dot",
            "multi_head_add",
            "multi_head_loc",
            "multi_head_multi_res_loc",
        ],
        help="Type of attention architecture",
    )
    group.add_argument(
        "--adim",
        default=320,
        type=int,
        help="Number of attention transformation dimensions",
    )
    group.add_argument(
        "--awin", default=5, type=int, help="Window size for location2d attention"
    )
    group.add_argument(
        "--aheads",
        default=4,
        type=int,
        help="Number of heads for multi head attention",
    )
    group.add_argument(
        "--aconv-chans",
        default=-1,
        type=int,
        help="Number of attention convolution channels \
                       (negative value indicates no location-aware attention)",
    )
    group.add_argument(
        "--aconv-filts",
        default=100,
        type=int,
        help="Number of attention convolution filters \
                       (negative value indicates no location-aware attention)",
    )
    group.add_argument(
        "--dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for the encoder",
    )
    return group
