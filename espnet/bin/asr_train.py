#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Automatic speech recognition model training script."""

import logging
import os
import random
import subprocess
import sys

from distutils.version import LooseVersion

import configargparse
import numpy as np
import torch

from espnet.utils.cli_utils import strtobool
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2")


# NOTE: you need this func to generate our sphinx doc
def get_parser(parser=None, required=True):
    """Get default arguments."""
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train an automatic speech recognition (ASR) model on one CPU, "
            "one or multiple GPUs",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="second config file path that overwrites the settings in `--config`.",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="third config file path that overwrites the settings in "
        "`--config` and `--config2`.",
    )

    parser.add_argument(
        "--ngpu",
        default=None,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--train-dtype",
        default="float32",
        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
        help="Data type for training (only pytorch backend). "
        "O0,O1,.. flags require apex. "
        "See https://nvidia.github.io/apex/amp.html#opt-levels",
    )
    parser.add_argument(
        "--backend",
        default="chainer",
        type=str,
        choices=["chainer", "pytorch"],
        help="Backend library",
    )
    parser.add_argument(
        "--outdir", type=str, required=required, help="Output directory"
    )
    parser.add_argument("--debugmode", default=1, type=int, help="Debugmode")
    parser.add_argument("--dict", required=required, help="Dictionary")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--debugdir", type=str, help="Output directory for debugging")
    parser.add_argument(
        "--resume",
        "-r",
        default="",
        nargs="?",
        help="Resume the training from snapshot",
    )
    parser.add_argument(
        "--resume-eps",
        default=True,
        type=strtobool,
        help="Resume adadelta epsilon from snapshot if True",
    )
    parser.add_argument(
        "--minibatches",
        "-N",
        type=int,
        default="-1",
        help="Process only N minibatches (for debug)",
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        type=str,
        nargs="?",
        help="Tensorboard log dir path",
    )
    parser.add_argument(
        "--report-interval-iters",
        default=100,
        type=int,
        help="Report interval iterations",
    )
    parser.add_argument(
        "--save-interval-iters",
        default=0,
        type=int,
        help="Save snapshot interval iterations",
    )
    # task related
    parser.add_argument(
        "--train-json",
        type=str,
        default=None,
        help="Filename of train label data (json)",
    )
    parser.add_argument(
        "--valid-json",
        type=str,
        default=None,
        help="Filename of validation label data (json)",
    )
    # network architecture
    parser.add_argument(
        "--model-module",
        type=str,
        default=None,
        help="model defined module (default: espnet.nets.xxx_backend.e2e_asr:E2E)",
    )
    # encoder
    parser.add_argument(
        "--num-encs", default=1, type=int, help="Number of encoders in the model."
    )
    # loss related
    parser.add_argument(
        "--ctc_type",
        default="warpctc",
        type=str,
        choices=["builtin", "warpctc", "gtnctc", "cudnnctc"],
        help="Type of CTC implementation to calculate loss.",
    )
    parser.add_argument(
        "--mtlalpha",
        default=0.5,
        type=float,
        help="Multitask learning coefficient, "
        "alpha: alpha*ctc_loss + (1-alpha)*att_loss ",
    )
    parser.add_argument(
        "--lsm-weight", default=0.0, type=float, help="Label smoothing weight"
    )
    # recognition options to compute CER/WER
    parser.add_argument(
        "--report-cer",
        default=False,
        action="store_true",
        help="Compute CER on development set",
    )
    parser.add_argument(
        "--report-wer",
        default=False,
        action="store_true",
        help="Compute WER on development set",
    )
    parser.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size")
    parser.add_argument("--penalty", default=0.0, type=float, help="Incertion penalty")
    parser.add_argument(
        "--maxlenratio",
        default=0.0,
        type=float,
        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""",
    )
    parser.add_argument(
        "--minlenratio",
        default=0.0,
        type=float,
        help="Input length ratio to obtain min output length",
    )
    parser.add_argument(
        "--maxioratio",
        default=float('inf'),
        type=float,
        help="Max ratio of input/output for training."
    )
    parser.add_argument(
        "--minioratio",
        default=0.0,
        type=float,
        help="Min ratio of input/output for training."
    )
    parser.add_argument(
        "--ctc-weight", default=0.3, type=float, help="CTC weight in joint decoding"
    )
    parser.add_argument(
        "--rnnlm", type=str, default=None, help="RNNLM model file to read"
    )
    parser.add_argument(
        "--rnnlm-conf", type=str, default=None, help="RNNLM model config file to read"
    )
    parser.add_argument("--lm-weight", default=0.1, type=float, help="RNNLM weight.")
    parser.add_argument("--sym-space", default="<space>", type=str, help="Space symbol")
    parser.add_argument("--sym-blank", default="<blank>", type=str, help="Blank symbol")
    parser.add_argument("--include-blank", default=False, type=strtobool,
                        help="Include blank in the output labels, for RNN-T / CTC.")
    # KB related
    parser.add_argument('--meetingKB', default=False, type=strtobool,
                       help='Use meeting level knowledge base')
    parser.add_argument('--meetingpath', type=str, default='',
                        help='The path to meeting level KB')
    parser.add_argument('--dictfile', type=str, default='',
                        help='Dictionary for word level LM')
    parser.add_argument('--lm-odim', type=int, default=0,
                        help='Language model output dimension')
    parser.add_argument('--attn-label', type=strtobool, default=False,
                        help='Load attention label for KB')
    parser.add_argument('--wordemb', default=0, type=int,
                        help='Number of word embedding hidden units')
    parser.add_argument('--PtrGen', default=False, type=strtobool,
                        help='use pointer generator for KB')
    parser.add_argument('--KBlextree', default=False, type=strtobool,
                        help='use lexicon tree to represent KB')
    parser.add_argument('--dynamicKBs', default=0, type=int,
                        help='Number of different KB arrangements')
    parser.add_argument('--PtrSche', default=0, type=int,
                        help='Pointer net scheduler')
    parser.add_argument('--PtrKBin', default=False, type=strtobool,
                        help='Use KB embedding as auxiliary input when PtrGen')
    parser.add_argument('--smoothprob', default=1.0, type=float,
                        help='KB probability smoothing factor')
    parser.add_argument('--attn_dim', default=1024, type=int,
                        help='KB lookup attention space')
    parser.add_argument('--KBlossfactor', default=0.0, type=float,
                        help='Factor of KB loss')
    parser.add_argument('--acousticonly', default=False, type=strtobool,
                        help='KB lookup only use acoustic information')
    parser.add_argument('--additive-attn', default=False, type=strtobool,
                        help='KB lookup uses additive attention')
    parser.add_argument('--ooKBemb', default=False, type=strtobool,
                        help='out-of-KB embedding for word piece')
    parser.add_argument('--KBmaxlen', default=500, type=int,
                        help='KB max sizes')
    parser.add_argument('--KBminlen', default=0, type=int,
                        help='KB min sizes')
    parser.add_argument('--DBmask', default=0.0, type=float,
                        help='Probability of masking from deep biasing')
    parser.add_argument('--DBdrop', default=0.0, type=float,
                        help='Probability of dropping bias entries')
    parser.add_argument('--DBinput', default=False, type=strtobool,
                        help='Use Deep biasing input')
    parser.add_argument("--randomKBsample", default=False, type=strtobool,
                        help="Organise KB randomly, set to False for decoding.")
    parser.add_argument("--curriculum", default=False, type=strtobool,
                        help="Use curriculum learning.")
    parser.add_argument('--fullepoch', default=0, type=int,
                        help='Epoch when curriculum ends')
    parser.add_argument('--unigram-file', default='', type=str,
                        help='File path of the unigram distribution of training set words')
    parser.add_argument('--mbrloss', default=False, type=strtobool,
                       help='Use MBR training')
    parser.add_argument('--mbrbeam', default=1, type=int,
                       help='MBR training beam size')
    parser.add_argument('--mbrnbest', default=1, type=int,
                       help='MBR training nbest size')
    parser.add_argument('--mbrlambda', default=0.0, type=float,
                       help='MBR training interpolation weight')
    parser.add_argument('--mbrGT', default=False, type=strtobool,
                       help='Include ground truth in MBR training')
    parser.add_argument('--mbrrareweight', default=0.0, type=float,
                       help='MBR training rare word weight')
    parser.add_argument('--mweweight', default=1.0, type=float,
                       help='MBR training MWE weight')
    parser.add_argument('--cfm-mbr-start', default=0, type=int,
                       help='MBR training start epoch for Conformer')
    parser.add_argument('--use-wp-errors', default=False, type=strtobool,
                       help='Use word piece errors for MBR training')
    parser.add_argument('--wrt-alignments', default=False, type=strtobool,
                       help='MBR loss differenciate wrt alignments')
    parser.add_argument('--treetype', default='', type=str,
                       help='Type of GNN to use')
    parser.add_argument('--treehid', default=0, type=int,
                       help='Tree network hidden state dims')
    parser.add_argument('--sampler', default=False, type=strtobool,
                       help='Use sequential sampler for training dataloader')
    parser.add_argument('--modalitymatch', default=False, type=strtobool,
                       help='Use modality matching finetuning')
    parser.add_argument('--robertadim', default=768, type=int,
                       help='Roberta model output dimension')
    parser.add_argument('--mmfactor', default=0.0, type=float,
                       help='Scaling factor for the MM loss')
    parser.add_argument('--pooling', default='', type=str,
                        help='pooling mechanism used for mm training')
    parser.add_argument('--doslu', default=False, type=strtobool,
                       help='Whether or not using SLU training')
    parser.add_argument('--intentfile', default='', type=str,
                        help='Path to the intent labels')
    parser.add_argument('--slotfile', default='', type=str,
                        help='Path to the slot labels')
    parser.add_argument('--slotfactor', default=0.0, type=float,
                       help='Scaling factor for the slot filling loss')
    parser.add_argument('--intentfactor', default=0.0, type=float,
                       help='Scaling factor for the intent detection loss')
    parser.add_argument('--wordlevel', default=False, type=strtobool,
                       help='Whether or not to perform wordlevel SLU')
    parser.add_argument('--jointrep', default=False, type=strtobool,
                       help='Whether or not to perform joint representation for SLU')
    parser.add_argument('--robertamask', default=0.0, type=float,
                       help='Mask fraction for the roberta encodings')
    parser.add_argument('--jointptrgen', default=False, type=strtobool,
                       help='Whether or not to include TCPGen probs in SLU')
    parser.add_argument('--slotKB', default=False, type=strtobool,
                       help='Whether or not to use slot prediction for KB')
    parser.add_argument('--ontology', default=None, type=str,
                        help='Path to the slot-value file')
    parser.add_argument('--topnslot', default=1, type=int,
                       help='Number of shortlisted slots')
    parser.add_argument('--graphslot', default=0.0, type=float,
                       help='use graph class posterior for slot filling')
    parser.add_argument('--stacked', default=False, type=strtobool,
                       help='Whether or not using stacked attention')
    parser.add_argument(
        "--init-roberta-model",
        default=None,
        type=str,
        help="Pre-trained ASR model to initialize all modules.",
    )
    
    # minibatch related
    parser.add_argument(
        "--sortagrad",
        default=0,
        type=int,
        nargs="?",
        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs",
    )
    parser.add_argument(
        "--batch-count",
        default="auto",
        choices=BATCH_COUNT_CHOICES,
        help="How to count batch_size. "
        "The default (auto) will find how to count by args.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch-seqs",
        "-b",
        default=0,
        type=int,
        help="Maximum seqs in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-bins",
        default=0,
        type=int,
        help="Maximum bins in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-in",
        default=0,
        type=int,
        help="Maximum input frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-out",
        default=0,
        type=int,
        help="Maximum output frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-inout",
        default=0,
        type=int,
        help="Maximum input+output frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--maxlen-in",
        "--batch-seq-maxlen-in",
        default=800,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the input sequence length > ML.",
    )
    parser.add_argument(
        "--maxlen-out",
        "--batch-seq-maxlen-out",
        default=150,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the output sequence length > ML",
    )
    parser.add_argument(
        "--n-iter-processes",
        default=0,
        type=int,
        help="Number of processes of iterator",
    )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        nargs="?",
        help="The configuration file for the pre-processing",
    )
    # optimization related
    parser.add_argument(
        "--opt",
        default="adadelta",
        type=str,
        choices=["adadelta", "adam", "noam"],
        help="Optimizer",
    )
    parser.add_argument(
        "--accum-grad", default=1, type=int, help="Number of gradient accumuration"
    )
    parser.add_argument(
        "--eps", default=1e-8, type=float, help="Epsilon constant for optimizer"
    )
    parser.add_argument(
        "--eps-decay", default=0.01, type=float, help="Decaying ratio of epsilon"
    )
    parser.add_argument(
        "--adam-lr", default=0.001, type=float, help="Adam learning rate"
    )
    parser.add_argument(
        "--adam-warmup-steps", default=0, type=int, help="Linear warmup steps for Adam LR"
    )
    parser.add_argument(
        "--weight-decay", default=0.0, type=float, help="Weight decay ratio"
    )
    parser.add_argument(
        "--criterion",
        default="acc",
        type=str,
        choices=["loss", "loss_eps_decay_only", "acc"],
        help="Criterion to perform epsilon decay",
    )
    parser.add_argument(
        "--threshold", default=1e-4, type=float, help="Threshold to stop iteration"
    )
    parser.add_argument(
        "--epochs", "-e", default=30, type=int, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--early-stop-criterion",
        default="validation/main/acc",
        type=str,
        nargs="?",
        help="Value to monitor to trigger an early stopping of the training",
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        nargs="?",
        help="Number of epochs to wait without improvement "
        "before stopping the training",
    )
    parser.add_argument(
        "--grad-clip", default=5, type=float, help="Gradient norm threshold to clip"
    )
    parser.add_argument(
        "--num-save-attention",
        default=3,
        type=int,
        help="Number of samples of attention to be saved",
    )
    parser.add_argument(
        "--num-save-ctc",
        default=3,
        type=int,
        help="Number of samples of CTC probability to be saved",
    )
    parser.add_argument(
        "--grad-noise",
        type=strtobool,
        default=False,
        help="The flag to switch to use noise injection to gradients during training",
    )
    parser.add_argument(
        "--weight-noise-std", default=0.0, type=float, help="The std of weight noise"
    )
    parser.add_argument(
        "--weight-noise-start",
        default=20000,
        type=int,
        help="The starting iteration of weight noise"
    )
    parser.add_argument(
        "--ema-decay",
        default=0.0,
        type=float,
        help="The decay value for exponential moving average"
    )
    # asr_mix related
    parser.add_argument(
        "--num-spkrs",
        default=1,
        type=int,
        choices=[1, 2],
        help="Number of speakers in the speech.",
    )
    # finetuning related
    parser.add_argument(
        "--enc-init",
        default=None,
        type=str,
        help="Pre-trained ASR model to initialize encoder.",
    )
    parser.add_argument(
        "--enc-init-mods",
        default="enc.enc.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules to initialize, separated by a comma.",
    )
    parser.add_argument(
        "--dec-init",
        default=None,
        type=str,
        help="Pre-trained ASR, MT or LM model to initialize decoder.",
    )
    parser.add_argument(
        "--dec-init-mods",
        default="att.,dec.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of decoder modules to initialize, separated by a comma.",
    )
    parser.add_argument(
        "--init-full-model",
        default=None,
        type=str,
        help="Pre-trained ASR model to initialize all modules.",
    )
    parser.add_argument(
        "--freeze-mods",
        default=None,
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of modules to freeze, separated by a comma.",
    )
    # front end related
    parser.add_argument(
        "--use-frontend",
        type=strtobool,
        default=False,
        help="The flag to switch to use frontend system.",
    )

    # WPE related
    parser.add_argument(
        "--use-wpe",
        type=strtobool,
        default=False,
        help="Apply Weighted Prediction Error",
    )
    parser.add_argument(
        "--wtype",
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
            "gru",
            "bgru",
            "grup",
            "bgrup",
            "vgggrup",
            "vggbgrup",
            "vgggru",
            "vggbgru",
        ],
        help="Type of encoder network architecture "
        "of the mask estimator for WPE. "
        "",
    )
    parser.add_argument("--wlayers", type=int, default=2, help="")
    parser.add_argument("--wunits", type=int, default=300, help="")
    parser.add_argument("--wprojs", type=int, default=300, help="")
    parser.add_argument("--wdropout-rate", type=float, default=0.0, help="")
    parser.add_argument("--wpe-taps", type=int, default=5, help="")
    parser.add_argument("--wpe-delay", type=int, default=3, help="")
    parser.add_argument(
        "--use-dnn-mask-for-wpe",
        type=strtobool,
        default=False,
        help="Use DNN to estimate the power spectrogram. "
        "This option is experimental.",
    )
    # Beamformer related
    parser.add_argument("--use-beamformer", type=strtobool, default=True, help="")
    parser.add_argument(
        "--btype",
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
            "gru",
            "bgru",
            "grup",
            "bgrup",
            "vgggrup",
            "vggbgrup",
            "vgggru",
            "vggbgru",
        ],
        help="Type of encoder network architecture "
        "of the mask estimator for Beamformer.",
    )
    parser.add_argument("--blayers", type=int, default=2, help="")
    parser.add_argument("--bunits", type=int, default=300, help="")
    parser.add_argument("--bprojs", type=int, default=300, help="")
    parser.add_argument("--badim", type=int, default=320, help="")
    parser.add_argument(
        "--bnmask",
        type=int,
        default=2,
        help="Number of beamforming masks, " "default is 2 for [speech, noise].",
    )
    parser.add_argument(
        "--ref-channel",
        type=int,
        default=-1,
        help="The reference channel used for beamformer. "
        "By default, the channel is estimated by DNN.",
    )
    parser.add_argument("--bdropout-rate", type=float, default=0.0, help="")
    # Feature transform: Normalization
    parser.add_argument(
        "--stats-file",
        type=str,
        default=None,
        help="The stats file for the feature normalization",
    )
    parser.add_argument(
        "--apply-uttmvn",
        type=strtobool,
        default=True,
        help="Apply utterance level mean " "variance normalization.",
    )
    parser.add_argument("--uttmvn-norm-means", type=strtobool, default=True, help="")
    parser.add_argument("--uttmvn-norm-vars", type=strtobool, default=False, help="")
    # Feature transform: Fbank
    parser.add_argument(
        "--fbank-fs",
        type=int,
        default=16000,
        help="The sample frequency used for " "the mel-fbank creation.",
    )
    parser.add_argument(
        "--n-mels", type=int, default=80, help="The number of mel-frequency bins."
    )
    parser.add_argument("--fbank-fmin", type=float, default=0.0, help="")
    parser.add_argument("--fbank-fmax", type=float, default=None, help="")
    return parser


def main(cmd_args):
    """Run the main training function."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    if args.backend == "chainer" and args.train_dtype != "float32":
        raise NotImplementedError(
            f"chainer backend does not support --train-dtype {args.train_dtype}."
            "Use --dtype float32."
        )
    if args.ngpu == 0 and args.train_dtype in ("O0", "O1", "O2", "O3", "float16"):
        raise ValueError(
            f"--train-dtype {args.train_dtype} does not support the CPU backend."
        )

    from espnet.utils.dynamic_import import dynamic_import

    if args.model_module is None:
        if args.num_spkrs == 1:
            model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
        else:
            model_module = "espnet.nets." + args.backend + "_backend.e2e_asr_mix:E2E"
    else:
        model_module = args.model_module
    model_class = dynamic_import(model_module)
    model_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)
    args.model_module = model_module
    if "chainer_backend" in args.model_module:
        args.backend = "chainer"
    if "pytorch_backend" in args.model_module:
        args.backend = "pytorch"

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    if args.ngpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            ngpu = len(cvd.split(","))
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
            try:
                p = subprocess.run(
                    ["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                ngpu = 0
            else:
                ngpu = len(p.stderr.decode().split("\n")) - 1
        args.ngpu = ngpu
    else:
        if is_torch_1_2_plus and args.ngpu != 1:
            logging.debug(
                "There are some bugs with multi-GPU processing in PyTorch 1.2+"
                + " (see https://github.com/pytorch/pytorch/issues/21108)"
            )
        ngpu = args.ngpu
    logging.info(f"ngpu: {ngpu}")

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # set random seed
    logging.info("random seed = %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, "rb") as f:
            dictionary = f.readlines()
        char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        if args.include_blank:
            char_list.insert(0, "<blank>")
            logging.info("RNN-T training, mtlalpha does not matter.")
        else:
            assert args.mtlalpha == 0, "The modified version has LAS only."
        char_list.append("<eos>")
        # for non-autoregressive maskctc model
        if "maskctc" in args.model_module:
            char_list.append("<mask>")
        args.char_list = char_list
    else:
        args.char_list = None

    # train
    logging.info("backend = " + args.backend)

    if args.num_spkrs == 1:
        if args.backend == "chainer":
            from espnet.asr.chainer_backend.asr import train

            train(args)
        elif args.backend == "pytorch":
            from espnet.asr.pytorch_backend.asr import train

            train(args)
        else:
            raise ValueError("Only chainer and pytorch are supported.")
    else:
        # FIXME(kamo): Support --model-module
        if args.backend == "pytorch":
            from espnet.asr.pytorch_backend.asr_mix import train

            train(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
