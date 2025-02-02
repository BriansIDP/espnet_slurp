#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""

import copy
import io
import json
import logging
import math
import os
import random
import sys

import yaml
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.nn.parallel import data_parallel

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import format_mulenc_args
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import set_weight_noise_on
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import freeze_modules
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.e2e_asr import pad_list
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets, get_meetings
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import EMA
from espnet.utils.training.train_utils import set_early_stop

import matplotlib

matplotlib.use("Agg")

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest


def _recursive_to(xs, device):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple):
        return tuple(_recursive_to(x, device) for x in xs)
    return xs


class CustomEvaluator(BaseEvaluator):
    """Custom Evaluator for Pytorch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        iterator (chainer.dataset.Iterator) : The train iterator.

        target (link | dict[str, link]) :Link object or a dictionary of
            links to evaluate. If this is just a link object, the link is
            registered by the name ``'main'``.

        device (torch.device): The device used.
        ngpu (int): The number of GPUs.
        use_ema (bool): Use EMA parameters for evaluation when available.

    """

    def __init__(self, model, iterator, target, device,
                 ngpu=None, use_ema=False):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.device = device
        if ngpu is not None:
            self.ngpu = ngpu
        elif device.type == "cpu":
            self.ngpu = 0
        else:
            self.ngpu = 1
        self.use_ema = use_ema
        if self.use_ema:
            assert hasattr(model, 'shadow'), "model does not have EMA params."
            self.default_name = 'validation(ema)'

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        """Main evaluate routine for CustomEvaluator."""
        iterator = self._iterators["main"]

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                x = _recursive_to(batch, self.device)
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    if self.ngpu <= 1:
                        if self.use_ema:
                            self.model.shadow(*x)
                        else:
                            self.model(*x)
                    else:
                        # apex does not support torch.nn.DataParallel
                        if self.use_ema:
                            data_parallel(self.model.shadow, x, range(self.ngpu))
                        else:
                            data_parallel(self.model, x, range(self.ngpu))

                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(StandardUpdater):
    """Custom Updater for Pytorch.

    Args:
        model (torch.nn.Module): The model to update.
        grad_clip_threshold (float): The gradient clipping value to use.
        train_iter (chainer.dataset.Iterator): The training iterator.
        optimizer (torch.optim.optimizer): The training optimizer.

        device (torch.device): The device to use.
        ngpu (int): The number of gpus to use.
        use_apex (bool): The flag to use Apex in backprop.
        weight_noise_std (float): The std of Gaussian weight noise.
        weight_noise_start (int): The starting iteration for weight noise.

    """

    def __init__(
        self,
        model,
        grad_clip_threshold,
        train_iter,
        optimizer,
        device,
        ngpu,
        grad_noise=False,
        accum_grad=1,
        use_apex=False,
        multi_pass=1,
        weight_noise_std=0.0,
        weight_noise_start=20000,
    ):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.device = device
        self.ngpu = ngpu
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.grad_noise = grad_noise
        self.iteration = 0
        self.use_apex = use_apex
        self.multi_pass = multi_pass
        self.weight_noise_std = weight_noise_std
        self.weight_noise_start = weight_noise_start
        self.weight_noise_on = False
        self.weight_noise_include = ("weight_ih", "weight_hh", "bias_ih", "bias_hh", "embed.weight", "output.weight", "output.bias")
        self.orig_params = {}

    def check_require_weight_noise(self, name):
        """Check if a parameter name matches any pattern in `self.weight_noise_include`."""
        require_wn = False
        for keyword in self.weight_noise_include:
            if keyword in name:
                require_wn = True
        return require_wn

    def add_weight_noise(self):
        """Add weight noise to model parameters by manipulating `param.data`."""
        if self.weight_noise_std > 0:
            # Weight noise is applied if either start_step has been reached
            # or after the first epoch that the model stops improving
            if self.iteration >= self.weight_noise_start or self.weight_noise_on:
                for name, param in self.model.named_parameters():
                    if self.check_require_weight_noise(name) and "shadow" not in name:
                        self.orig_params[name] = param.clone()
                        noise = self.weight_noise_std * torch.randn(param.size()).to(param.device)
                        param.data = param.data + noise

    def restore_params(self):
        """Restore parameters to the ones before addtive noise."""
        if self.orig_params:
            for name, param in self.model.named_parameters():
                if name in self.orig_params:
                    param.data = self.orig_params[name].data
            self.orig_params = {}

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        epoch = train_iter.epoch
        # gs534 - synchronize epoch number
        self.model.dec.epoch = epoch

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        # self.iteration += 1 # Increase may result in early report,
        # which is done in other place automatically.
        x = _recursive_to(batch, self.device)
        is_new_epoch = train_iter.epoch != epoch
        # When the last minibatch in the current epoch is given,
        # gradient accumulation is turned off in order to evaluate the model
        # on the validation set in every epoch.
        # see details in https://github.com/espnet/espnet/pull/1388

        # Drop minibatch if the batchsize is 1
        if x[0].shape[0] <= 1:
            return

        # Add Gaussian noise on weights before each forward computation
        if self.forward_count == 0:
            self.add_weight_noise()

        if len(x[0].shape) == 4:
            num_shifts = x[0].shape[-1]
            for i in range(num_shifts):
                feat = (x[0][:, :, :, i], x[1], x[2])
                # Compute the loss at this time step and accumulate it
                if self.ngpu <= 1:
                    loss = self.model(*feat).mean() / float(self.accum_grad)
                else:
                    # apex does not support torch.nn.DataParallel
                    loss = (
                        data_parallel(self.model, feat, range(self.ngpu)).mean() / self.accum_grad
                    )
                if self.use_apex:
                    from apex import amp

                    # NOTE: for a compatibility with noam optimizer
                    opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
        else:
            assert len(x[0].shape) == 3
            # Compute the loss at this time step and accumulate it
            if self.ngpu <= 1:
                loss = self.model(*x).mean() / self.accum_grad
            else:
                # apex does not support torch.nn.DataParallel
                loss = (
                    data_parallel(self.model, x, range(self.ngpu)).mean() / self.accum_grad
                )
            if self.use_apex:
                from apex import amp

                # NOTE: for a compatibility with noam optimizer
                opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        # gradient noise injection
        if self.grad_noise:
            from espnet.asr.asr_utils import add_gradient_noise

            add_gradient_noise(
                self.model, self.iteration, duration=100, eta=1.0, scale_factor=0.55
            )

        # update parameters
        self.forward_count += 1
        if not is_new_epoch and self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # reduce the clipping threshold to 1/10 when weight noise is applied
        if self.weight_noise_std and (self.iteration >= self.weight_noise_start or
                                      self.weight_noise_on):
            clip_threshold = self.grad_clip_threshold / 10
        else:
            clip_threshold = self.grad_clip_threshold
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), clip_threshold
        )
        logging.info("grad norm={}".format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning("grad norm is nan. Do not update model.")
        else:
            # restore parameters to exclude weight noise before update
            self.restore_params()
            optimizer.step()
            if hasattr(self.model, 'decay'):
                self.model.update_ema()
        optimizer.zero_grad()

    def update(self):
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        if self.forward_count == 0:
            self.iteration += 1


class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32, getmeeting=False, gettext=False, getslu=False,
                 getprevtext=False):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype
        self.getmeeting = getmeeting
        self.gettext = gettext
        self.getprevtext = getprevtext
        self.getslu = getslu

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0][:2]
        meetings = None
        orig_text = None
        prev_text = None
        slu_intents = None
        slu_slots = None
        if not xs:
            logging.info("Empty batch.")
            xs_pad = torch.tensor([]).to(device)
            ilens = torch.tensor([]).to(device)
            ys_pad = torch.tensor([]).to(device)
            return xs_pad, ilens, ys_pad, meetings

        itemcount = 2
        if self.getmeeting and len(batch[0]) > itemcount:
            meetings = batch[0][itemcount]
            itemcount += 1
        if self.gettext and len(batch[0]) > itemcount:
            orig_text = batch[0][itemcount]
            itemcount += 1
            if self.getprevtext and len(batch[0]) > itemcount:
                prev_text = batch[0][itemcount]
                itemcount += 1
        if self.getslu and len(batch[0]) > itemcount:
            slu_intents = batch[0][itemcount]
            itemcount += 1
            slu_slots = batch[0][itemcount]
            itemcount += 1

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[:: self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == "c":
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {"real": xs_pad_real, "imag": xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(
                device, dtype=self.dtype
            )

        ilens = torch.from_numpy(ilens).to(device)
        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(
                    np.array(y[0][:]) if isinstance(y, tuple) else y
                ).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_pad, ilens, ys_pad, meetings, orig_text, slu_intents, slu_slots, prev_text


class CustomConverterMulEnc(object):
    """Custom batch converter for Pytorch in multi-encoder case.

    Args:
        subsampling_factors (list): List of subsampling factors for each encoder.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsamping_factors=[1, 1], dtype=torch.float32):
        """Initialize the converter."""
        self.subsamping_factors = subsamping_factors
        self.ignore_id = -1
        self.dtype = dtype
        self.num_encs = len(subsamping_factors)

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple( list(torch.Tensor), list(torch.Tensor), torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs_list = batch[0][: self.num_encs]
        ys = batch[0][-1]

        # perform subsampling
        if np.sum(self.subsamping_factors) > self.num_encs:
            xs_list = [
                [x[:: self.subsampling_factors[i], :] for x in xs_list[i]]
                for i in range(self.num_encs)
            ]

        # get batch of lengths of input sequences
        ilens_list = [
            np.array([x.shape[0] for x in xs_list[i]]) for i in range(self.num_encs)
        ]

        # perform padding and convert to tensor
        # currently only support real number
        xs_list_pad = [
            pad_list([torch.from_numpy(x).float() for x in xs_list[i]], 0).to(
                device, dtype=self.dtype
            )
            for i in range(self.num_encs)
        ]

        ilens_list = [
            torch.from_numpy(ilens_list[i]).to(device) for i in range(self.num_encs)
        ]
        # NOTE: this is for multi-task learning (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(np.array(y[0]) if isinstance(y, tuple) else y).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_list_pad, ilens_list, ys_pad


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    if args.num_encs > 1:
        args = format_mulenc_args(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get input and output dimension info
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    idim_list = [
        int(valid_json[utts[0]]["input"][i]["shape"][-1]) for i in range(args.num_encs)
    ]
    if args.preprocess_conf is not None:
        with io.open(args.preprocess_conf, encoding="utf-8") as f:
            preprocess_conf = yaml.safe_load(f)
        for process in preprocess_conf["process"]:
            if process["type"] == "subsample" and process["shift"] is True:
                idim_list = [int(i * process["sample_rate"]) for i in idim_list]
                break
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])
    for i in range(args.num_encs):
        logging.info("stream{}: input dims : {}".format(i + 1, idim_list[i]))
    logging.info("#output dims: " + str(odim))

    # specify attention, CTC, hybrid mode
    if "transducer" in args.model_module or 'rnnt' in args.model_module:
        if (
            getattr(args, "etype", False) == "transformer"
            or getattr(args, "dtype", False) == "transformer"
        ):
            mtl_mode = "transformer_transducer"
        else:
            mtl_mode = "transducer"
        logging.info("Pure transducer mode")
    elif args.mtlalpha == 1.0:
        mtl_mode = "ctc"
        logging.info("Pure CTC mode")
    elif args.mtlalpha == 0.0:
        mtl_mode = "att"
        logging.info("Pure attention mode")
    else:
        mtl_mode = "mtl"
        logging.info("Multitask learning mode")

    if (args.enc_init is not None or args.dec_init is not None) and args.num_encs == 1:
        model = load_trained_modules(idim_list[0], odim, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(
            idim_list[0] if args.num_encs == 1 else idim_list, odim, args
        )
    assert isinstance(model, ASRInterface)
    total_subsampling_factor = model.get_total_subsampling_factor()

    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(args.char_list), rnnlm_args.layer, rnnlm_args.unit)
        )
        torch_load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (idim_list[0] if args.num_encs == 1 else idim_list, odim, vars(args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )
    for key in sorted(vars(args).keys()):
        logging.info("ARGS: " + key + ": " + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.warning(
                "batch size is automatically increased (%d -> %d)"
                % (args.batch_size, args.batch_size * args.ngpu)
            )
            args.batch_size *= args.ngpu
        if args.num_encs > 1:
            # TODO(ruizhili): implement data parallel for multi-encoder setup.
            raise NotImplementedError(
                "Data parallel is not supported for multi-encoder setup."
            )

    # set torch device
    print('device count: {}'.format(torch.cuda.device_count()))
    print('current device')
    print(torch.cuda.current_device())
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    if args.freeze_mods:
        model, model_params = freeze_modules(model, args.freeze_mods)
    else:
        model_params = model.parameters()

    logging.warning(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )

    # Setup an optimizer
    if args.opt == "adadelta":
        optimizer = torch.optim.Adadelta(
            model_params, rho=0.95, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.opt == "adam":
        from espnet.nets.pytorch_backend.rnn.optimizer import get_std_opt
        optimizer = get_std_opt(model_params, args.adam_warmup_steps, args.adam_lr,
                                args.weight_decay)
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

        # For transformer-transducer, adim declaration is within the block definition.
        # Thus, we need retrieve the most dominant value (d_hidden) for Noam scheduler.
        if hasattr(args, "enc_block_arch") or hasattr(args, "dec_block_arch"):
            adim = model.most_dom_dim
        else:
            adim = getattr(args, 'adim')

        optimizer = get_std_opt(
            model_params, adim, args.transformer_warmup_steps, args.transformer_lr
        )
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(
                f"You need to install apex for --train-dtype {args.train_dtype}. "
                "See https://github.com/NVIDIA/apex#linux"
            )
            raise e
        if args.opt == "noam":
            model, optimizer.optimizer = amp.initialize(
                model, optimizer.optimizer, opt_level=args.train_dtype
            )
        else:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.train_dtype
            )
        use_apex = True

        from espnet.nets.pytorch_backend.ctc import CTC

        amp.register_float_function(CTC, "loss_fn")
        amp.init()
        logging.warning("register ctc as float function")
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup EMA
    if args.ema_decay > 0:
        model = EMA(model, args.ema_decay)

    # Setup a converter
    if args.num_encs == 1:
        converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype,
                                    getmeeting=args.meetingKB, gettext=(args.modalitymatch or args.jointrep),
                                    getslu=args.doslu, getprevtext=args.gethistory)
    else:
        converter = CustomConverterMulEnc(
            [i[0] for i in model.subsample_list], dtype=dtype
        )

    # read json data
    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    valid = make_batchset(
        valid_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )

    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
        maxioratio=args.maxioratio,
        minioratio=args.minioratio,
        getmeeting=args.meetingKB,
        gettext=(args.modalitymatch or args.jointrep),
        getprevtext=args.gethistory,
        getslu=args.doslu,
    )
    load_cv = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
        getmeeting=args.meetingKB,
        gettext=(args.modalitymatch or args.jointrep),
        getprevtext=args.gethistory,
        getslu=args.doslu,
    )
    # fix random seeds for data loader
    torch.manual_seed(args.seed)
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter = ChainerDataLoader(
        dataset=TransformDataset(train, lambda data: converter([load_tr(data)])),
        batch_size=1,
        num_workers=args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
        sampler=getattr(args, 'sampler', False)
    )
    valid_iter = ChainerDataLoader(
        dataset=TransformDataset(valid, lambda data: converter([load_cv(data)])),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=args.n_iter_processes,
    )

    # Set up a trainer
    updater = CustomUpdater(
        model,
        args.grad_clip,
        {"main": train_iter},
        optimizer,
        device,
        args.ngpu,
        args.grad_noise,
        args.accum_grad,
        use_apex=use_apex,
        weight_noise_std=args.weight_noise_std,
        weight_noise_start=args.weight_noise_start,
    )
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.outdir)

    if use_sortagrad:
        trainer.extend(
            ShufflingEnabler([train_iter]),
            trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, "epoch"),
        )

    # gs534 - lextree
    if args.init_full_model is not None and args.init_full_model != '':
        snapshot_dict = torch.load(args.init_full_model, map_location=lambda storage, loc: storage)
        trainer.updater.model.load_state_dict(snapshot_dict, strict=False)
        if getattr(trainer.updater.model, 'slunet', False) and getattr(trainer.updater.model.slunet, 'gcn_l1', False):
            if getattr(trainer.updater.model.dec, 'gcn_l1', False):
                trainer.updater.model.slunet.gcn_l1.weight.data.copy_(trainer.updater.model.dec.gcn_l1.weight.data)
                trainer.updater.model.slunet.gcn_l1.bias.data.copy_(trainer.updater.model.dec.gcn_l1.bias.data)
                trainer.updater.model.slunet.gcn_l2.weight.data.copy_(trainer.updater.model.dec.gcn_l2.weight.data)
                trainer.updater.model.slunet.gcn_l2.bias.data.copy_(trainer.updater.model.dec.gcn_l2.bias.data)

    # Resume from a snapshot
    if args.resume:
        logging.info("resumed from %s" % args.resume)
        torch_resume(args.resume, trainer, strict=(not args.meetingKB))
        # reset adadelta eps if resume_eps is False
        if not args.resume_eps and args.opt == "adadelta":
            trainer.updater.get_optimizer("main").param_groups[0]["eps"] = args.eps

    # Evaluate the model with the test dataset for each epoch
    if args.save_interval_iters > 0:
        trigger = (args.save_interval_iters, "iteration")
    else:
        trigger = None
    trainer.extend(
        CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu, use_ema=False),
        trigger=trigger,
    )
    # Have a separate evaluator for EMA, the name will be 'validation(ema)/'
    if args.ema_decay > 0:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu, use_ema=True),
            trigger=trigger,
        )

    # Save attention weight each epoch
    is_attn_plot = (
        "transformer" in args.model_module
        or "conformer" in args.model_module
        or mtl_mode in ["att", "mtl"]
    ) or mtl_mode == "transformer_transducer"

    if args.num_save_attention > 0 and is_attn_plot:
        data = sorted(
            list(valid_json.items())[: args.num_save_attention],
            key=lambda x: int(x[1]["input"][0]["shape"][1]),
            reverse=True,
        )
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn,
            data,
            args.outdir + "/att_ws",
            converter=converter,
            transform=load_cv,
            device=device,
            subsampling_factor=total_subsampling_factor,
        )
        trainer.extend(att_reporter, trigger=(1, "epoch"))
    else:
        att_reporter = None

    # Save CTC prob at each epoch
    if mtl_mode in ["ctc", "mtl"] and args.num_save_ctc > 0:
        # NOTE: sort it by output lengths
        data = sorted(
            list(valid_json.items())[: args.num_save_ctc],
            key=lambda x: int(x[1]["output"][0]["shape"][0]),
            reverse=True,
        )
        if hasattr(model, "module"):
            ctc_vis_fn = model.module.calculate_all_ctc_probs
            plot_class = model.module.ctc_plot_class
        else:
            ctc_vis_fn = model.calculate_all_ctc_probs
            plot_class = model.ctc_plot_class
        ctc_reporter = plot_class(
            ctc_vis_fn,
            data,
            args.outdir + "/ctc_prob",
            converter=converter,
            transform=load_cv,
            device=device,
            subsampling_factor=total_subsampling_factor,
        )
        trainer.extend(ctc_reporter, trigger=(1, "epoch"))
    else:
        ctc_reporter = None

    # Make a plot for training and validation values
    if args.num_encs > 1:
        report_keys_loss_ctc = [
            "main/loss_ctc{}".format(i + 1) for i in range(model.num_encs)
        ] + ["validation/main/loss_ctc{}".format(i + 1) for i in range(model.num_encs)]
        report_keys_cer_ctc = [
            "main/cer_ctc{}".format(i + 1) for i in range(model.num_encs)
        ] + ["validation/main/cer_ctc{}".format(i + 1) for i in range(model.num_encs)]
    if args.ema_decay > 0:
        report_keys_loss_ema = ['validation(ema)/main/loss',
                                'validation(ema)/main/loss_ctc',
                                'validation(ema)/main/loss_att']
        report_keys_cer_ema = ['validation(ema)/main/cer_ctc']
        report_keys_acc_ema = ['validation(ema)/main/acc']
    trainer.extend(
        extensions.PlotReport(
            [
                "main/loss",
                "validation/main/loss",
                "main/loss_ctc",
                "validation/main/loss_ctc",
                "main/loss_att",
                "validation/main/loss_att",
            ]
            + ([] if args.num_encs == 1 else report_keys_loss_ctc)
            + ([] if args.ema_decay == 0 else report_keys_loss_ema),
            "epoch",
            file_name="loss.png",
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/acc", "validation/main/acc"]
            + ([] if args.ema_decay == 0 else report_keys_acc_ema),
            "epoch",
            file_name="acc.png"
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/cer_ctc", "validation/main/cer_ctc"]
            + ([] if args.num_encs == 1 else report_keys_cer_ctc)
            + ([] if args.ema_decay == 0 else report_keys_cer_ema),
            "epoch",
            file_name="cer.png",
        )
    )

    # Save best models based on original params
    trainer.extend(
        snapshot_object(model, "model.loss.best"),
        trigger=training.triggers.MinValueTrigger("validation/main/loss"),
    )
    if mtl_mode not in ["ctc", "transducer", "transformer_transducer"]:
        trainer.extend(
            snapshot_object(model, "model.acc.best"),
            trigger=training.triggers.MaxValueTrigger("validation/main/acc"),
        )
    # Save best models based on EMA params
    if args.ema_decay > 0:
        trainer.extend(
            snapshot_object(model, "model.ema.loss.best"),
            trigger=training.triggers.MinValueTrigger("validation(ema)/main/loss"),
        )
        if mtl_mode not in ["ctc", "transducer", "transformer_transducer"]:
            trainer.extend(
                snapshot_object(model, "model.ema.acc.best"),
                trigger=training.triggers.MaxValueTrigger("validation(ema)/main/acc"),
            )

    # save snapshot which contains model and optimizer states
    if args.save_interval_iters > 0:
        trainer.extend(
            torch_snapshot(filename="snapshot.iter.{.updater.iteration}"),
            trigger=(args.save_interval_iters, "iteration"),
        )

    # save snapshot at every epoch - for model averaging
    trainer.extend(torch_snapshot(), trigger=(1, "epoch"))

    # Turn on weight noise once the validation acc or loss stops improving
    # Normally, eps_decay is set to 1 if weight noise is used
    if args.weight_noise_std > 0:
        if args.criterion == 'acc' and mtl_mode != "ctc":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + '/model.acc.best', load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
            trainer.extend(
                set_weight_noise_on(),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
        elif args.criterion == "loss":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.loss.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
            trainer.extend(
                set_weight_noise_on(),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )

    # epsilon decay in the optimizer
    # NOTE: epsilon decay conditions on the validation criterion using original parameters
    if args.opt == "adadelta":
        if args.criterion == "acc" and mtl_mode != "ctc":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.acc.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
        elif args.criterion == "loss":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.loss.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
        # NOTE: In some cases, it may take more than one epoch for the model's loss
        # to escape from a local minimum.
        # Thus, restore_snapshot extension is not used here.
        # see details in https://github.com/espnet/espnet/pull/2171
        elif args.criterion == "loss_eps_decay_only":
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )

    # Write a log of evaluation statistics for each epoch
    trainer.extend(
        extensions.LogReport(trigger=(args.report_interval_iters, "iteration"))
    )
    report_keys = [
        "epoch",
        "iteration",
        "main/loss",
        "main/loss_ctc",
        "main/loss_att",
        "main/loss_MBR",
        "validation/main/loss",
        "validation/main/loss_ctc",
        "validation/main/loss_att",
        "main/acc",
        "validation/main/acc",
        "main/cer_ctc",
        "validation/main/cer_ctc",
        "elapsed_time",
    ]
    if args.num_encs > 1:
        report_keys += (report_keys_cer_ctc + report_keys_loss_ctc)
    if args.ema_decay > 0:
        report_keys += (report_keys_loss_ema + report_keys_acc_ema + report_keys_cer_ema)
    if args.opt == "adadelta":
        trainer.extend(
            extensions.observe_value(
                "eps",
                lambda trainer: trainer.updater.get_optimizer("main").param_groups[0][
                    "eps"
                ],
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
        report_keys.append("eps")
    elif args.opt == 'adam' or args.opt == 'noam':
        trainer.extend(
            extensions.observe_value(
                "lr",
                lambda trainer: trainer.updater.get_optimizer("main").param_groups[0][
                    "lr"
                ],
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
        report_keys.append("lr")
    if args.report_cer:
        report_keys.append("validation/main/cer")
        if args.ema_decay > 0:
            report_keys.append("validation(ema)/main/cer")
    if args.report_wer:
        report_keys.append("validation/main/wer")
        if args.ema_decay > 0:
            report_keys.append("validation(ema)/main/wer")
    trainer.extend(
        extensions.PrintReport(report_keys),
        trigger=(args.report_interval_iters, "iteration"),
    )

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(
            TensorboardLogger(
                SummaryWriter(args.tensorboard_dir),
                att_reporter=att_reporter,
                ctc_reporter=ctc_reporter,
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model, args.use_ema, training=False)
    assert isinstance(model, ASRInterface)
    model.recog_args = args

    if args.streaming_mode and "transformer" in train_args.model_module:
        raise NotImplementedError("streaming mode for transformer is not implemented")
    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    # read rnnlm
    if args.rnnlm:
        # rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        # if getattr(rnnlm_args, "model_module", "default") != "default":
        #     raise ValueError(
        #         "use '--api v2' option to decode with non-default language model"
        #     )
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list) - (1 if '<blank>' in train_args.char_list else 0),
                args.rnnlm_layer,
                args.rnnlm_unit,
                getattr(args, "embed_unit", None),  # for backward compatibility
            )
        )
        if args.external:
            # load the state_dict for the main LM
            ext_rnnlm = torch.load(args.rnnlm, map_location=lambda storage, loc: storage)
            ext_rnnlm_dict = ext_rnnlm.state_dict()
            # fill in state dicts individually
            for name, param in rnnlm.state_dict().items():
                # Default LM has lstm as a model list
                if 'rnn' in name:
                    _, param_name, layer, param_kind = name.split('.')
                    param_name = '.'.join([param_name, param_kind+'_l{}'.format(layer)])
                elif 'embed' in name:
                    param_name = '.'.join(['encoder', name.split('.')[-1]])
                elif 'lo' in name:
                    param_name = '.'.join(['decoder', name.split('.')[-1]])
                else:
                    param_name = '.'.join(name.split('.')[1:])
                param = ext_rnnlm_dict[param_name].data
                rnnlm.state_dict()[name].copy_(param)
        else:
            torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    # LM estimation and cancellation
    est_lm = None
    if args.est_lm:
        est_lm = torch.load(args.est_lm, map_location=lambda storage, loc: storage)
        est_lm.eval()

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(word_dict),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(
                    word_rnnlm.predictor, rnnlm.predictor, word_dict, char_dict
                )
            )
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(
                    word_rnnlm.predictor, word_dict, char_dict
                )
            )

    # gs534 - LM fusion
    best_fusion = None
    best_est = None

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # gs534 - LM for KB selection
    sel_lm = None
    best_hid = None
    best_fusion = None
    unigram_penalty = None
    dial_history = '' if args.usehistory else None
    biasinghist = []
    laststate = '' if args.getlaststate else None
    if args.select:
        sel_lm = torch.load(args.sel_lm, map_location=lambda storage, loc: storage)
        sel_lm.eval()
    if (args.select and args.classlm) or args.slotlist:
        model.meeting_KB.get_tree_from_classes(args.classfile, args.classorder, entity=args.entity)
        # pre-compute GNN tree encodings
        with torch.no_grad():
            for i in range(len(model.meeting_KB.classtrees)):
                print("Encoding tree for class {}".format(i))
                model.eval()
                if model.meeting_KB.classtrees[i][0] != {}:
                    model.dec.encode_tree(model.meeting_KB.classtrees[i])
        if args.unigram != '':
            unigram_penalty = []
            with open(args.unigram) as fin:
                unigram = json.load(fin)
            for classname in model.meeting_KB.classorder:
                count = 0
                words = model.meeting_KB.classdict[classname]
                for word in words:
                    count += unigram[word] if word in unigram else 0
                unigram_penalty.append(max(1, count))
            unigram_penalty = torch.Tensor(unigram_penalty)
            unigram_penalty = torch.log(unigram_penalty / unigram_penalty.sum()) * args.penalty_factor

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    # select subset
    keys = list(js.keys())
    if args.subset_percentage < 1.0:
        keys = random.sample(keys, int(len(keys) * args.subset_percentage))
        logging.warning("Only %f%% of utterances are decoded" % (args.subset_percentage * 100))
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    # load transducer beam search
    if hasattr(model, "is_rnnt"):
        if hasattr(model, "dec"):
            trans_decoder = model.dec
        else:
            trans_decoder = model.decoder
        joint_network = model.joint_network
        KBmodules = {'meeting_KB': None, 'DBinput': False, 'PtrGen': False}
        if model.meeting_KB is not None:
            KBmodules['meeting_KB'] = model.meeting_KB
            KBmodules['char_list'] = train_args.char_list
            if model.DBinput:
                KBmodules['DBembed'] = model.DBembed
                KBmodules['DBinput'] = True
            if model.PtrGen:
                KBmodules['PtrGen'] = True
                KBmodules['ooKBemb'] = model.ooKBemb
                KBmodules['Qproj_char'] = model.Qproj_char
                KBmodules['Qproj_acoustic'] = model.Qproj_acoustic
                KBmodules['Kproj'] = model.Kproj
                KBmodules['pointer_gate'] = model.pointer_gate
                KBmodules['smoothprob'] = getattr(model, 'smoothprob', 1.0)
                KBmodules['KBin'] = getattr(model, 'KBin', False)
                KBmodules['treetype'] = getattr(model, 'treetype', '')
                if getattr(model, 'treetype', '').startswith('gcn'):
                    KBmodules['gcn_l1'] = model.gcn_l1
                    KBmodules['gcn_l2'] = model.gcn_l2
                    if hasattr(model, 'gcn_l3'):
                        KBmodules['gcn_l3'] = model.gcn_l3
                elif getattr(model, 'treetype', '') == '':
                    KBmodules['recursive_proj'] = model.recursive_proj
                KBmodules['tree_hid'] = getattr(model, 'tree_hid', '')

        beam_search_transducer = BeamSearchTransducer(
            decoder=trans_decoder,
            joint_network=joint_network,
            beam_size=args.beam_size,
            nbest=args.nbest,
            lm=rnnlm,
            lm_weight=args.lm_weight,
            search_type=args.search_type,
            max_sym_exp=args.max_sym_exp,
            u_max=args.u_max,
            nstep=args.nstep,
            prefix_alpha=args.prefix_alpha,
            score_norm=args.score_norm,
            KBmodules=KBmodules,
            sel_lm=sel_lm,
            topk=args.topk
        )

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(keys, 1):
                logging.info("(%d/%d) decoding " + name, idx, len(keys))
                batch = [(name, js[name])]
                # gs534 - directly get meeting KB for it
                if getattr(train_args, 'randomKBsample', False) and getattr(train_args, 'dynamicKBs', 0) == 0:
                    meeting = set([tuple(word) for word in js[name]['output'][0]['uttKB']])
                else:
                    meeting = get_meetings([name])
                # Slot KB prediction
                slotlist = []
                if args.slotlist:
                    slotlist = js[name]['output'][0]['shortlist']
                # system response
                if args.usehistory and 'prevutts' in js[name]['output'][0]:
                    if js[name]['output'][0]['prevutts'] == '':
                        dial_history = ''
                        laststate = ''
                        biasinghist = []
                    else:
                        dial_history += js[name]['output'][0]['prevutts']
                # oracle text
                oracletext = None
                if args.oracletext:
                    oracletext = [int(idx) for idx in js[name]['output'][0]['tokenid'].split()]

                feat = load_inputs_and_targets(batch)
                feat = (
                    feat[0][0]
                    if args.num_encs == 1
                    else [feat[idx][0] for idx in range(model.num_encs)]
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with window size %d frames",
                        args.streaming_window,
                    )
                    se2e = WindowStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    for i in range(0, feat.shape[0], args.streaming_window):
                        logging.info(
                            "Feeding frames %d - %d", i, i + args.streaming_window
                        )
                        se2e.accept_input(feat[i : i + args.streaming_window])
                    logging.info("Running offline attention decoder")
                    se2e.decode_with_attention_offline()
                    logging.info("Offline attention decoder finished")
                    nbest_hyps = se2e.retrieve_recognition()
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with threshold value %d",
                        args.streaming_min_blank_dur,
                    )
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i : i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                elif hasattr(model, "is_rnnt"):
                    nbest_hyps = model.recognize(feat, beam_search_transducer, meetings=meeting,
                        estlm=est_lm, estlm_factor=args.estlm_factor, prev_hid=best_hid)
                    if args.select:
                        best_hid = nbest_hyps[0]['prev_hid']
                    # logging.info('Trace here {}'.format(' '.join([str(t) for t in nbest_hyps[0]['trace']])))
                else:
                    if not args.crossutt:
                        best_fusion, best_est = None, None
                    history_to_send = dial_history + laststate + '\n' if args.getlaststate and laststate != '' else dial_history
                    nbest_hyps, best_fusion, best_est, best_hid, laststate = model.recognize(
                        feat, args, train_args.char_list, rnnlm, meetings=meeting,
                        best_fusion=best_fusion, estlm=est_lm, estlm_factor=args.estlm_factor,
                        best_est=best_est, sel_lm=sel_lm, prev_hid=best_hid, slotlist=slotlist,
                        oracletext=oracletext, unigram=unigram_penalty, history=history_to_send, bhist=biasinghist
                    )
                    if args.usehistory and dial_history is not None:
                        dial_history += nbest_hyps[0]['yseq_str'].lower()
                        # biasinghist = nbest_hyps[0]['entities']
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, train_args.char_list
                )

    else:

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        if args.batchsize > 1:
            feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = (
                    load_inputs_and_targets(batch)[0]
                    if args.num_encs == 1
                    else load_inputs_and_targets(batch)
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    raise NotImplementedError
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    if args.batchsize > 1:
                        raise NotImplementedError
                    feat = feats[0]
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i : i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                    nbest_hyps = [nbest_hyps]
                else:
                    nbest_hyps = model.recognize_batch(
                        feats, args, train_args.char_list, rnnlm=rnnlm
                    )

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(
                        js[name], nbest_hyp, train_args.char_list
                    )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )


def enhance(args):
    """Dumping enhanced speech and mask.

    Args:
        args (namespace): The program arguments.
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # TODO(ruizhili): implement enhance for multi-encoder model
    assert args.num_encs == 1, "number of encoder should be 1 ({} is given)".format(
        args.num_encs
    )

    # load trained model parameters
    logging.info("reading model parameters from " + args.model)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, ASRInterface)
    torch_load(args.model, model)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=None,  # Apply pre_process in outer func
    )
    if args.batchsize == 0:
        args.batchsize = 1

    # Creates writers for outputs from the network
    if args.enh_wspecifier is not None:
        enh_writer = file_writer_helper(args.enh_wspecifier, filetype=args.enh_filetype)
    else:
        enh_writer = None

    # Creates a Transformation instance
    preprocess_conf = (
        train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf
    )
    if preprocess_conf is not None:
        logging.info(f"Use preprocessing: {preprocess_conf}")
        transform = Transformation(preprocess_conf)
    else:
        transform = None

    # Creates a IStft instance
    istft = None
    frame_shift = args.istft_n_shift  # Used for plot the spectrogram
    if args.apply_istft:
        if preprocess_conf is not None:
            # Read the conffile and find stft setting
            with open(preprocess_conf) as f:
                # Json format: e.g.
                #    {"process": [{"type": "stft",
                #                  "win_length": 400,
                #                  "n_fft": 512, "n_shift": 160,
                #                  "window": "han"},
                #                 {"type": "foo", ...}, ...]}
                conf = json.load(f)
                assert "process" in conf, conf
                # Find stft setting
                for p in conf["process"]:
                    if p["type"] == "stft":
                        istft = IStft(
                            win_length=p["win_length"],
                            n_shift=p["n_shift"],
                            window=p.get("window", "hann"),
                        )
                        logging.info(
                            "stft is found in {}. "
                            "Setting istft config from it\n{}".format(
                                preprocess_conf, istft
                            )
                        )
                        frame_shift = p["n_shift"]
                        break
        if istft is None:
            # Set from command line arguments
            istft = IStft(
                win_length=args.istft_win_length,
                n_shift=args.istft_n_shift,
                window=args.istft_window,
            )
            logging.info(
                "Setting istft config from the command line args\n{}".format(istft)
            )

    # sort data
    keys = list(js.keys())
    feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
    sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
    keys = [keys[i] for i in sorted_index]

    def grouper(n, iterable, fillvalue=None):
        kargs = [iter(iterable)] * n
        return zip_longest(*kargs, fillvalue=fillvalue)

    num_images = 0
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    for names in grouper(args.batchsize, keys, None):
        batch = [(name, js[name]) for name in names]

        # May be in time region: (Batch, [Time, Channel])
        org_feats = load_inputs_and_targets(batch)[0]
        if transform is not None:
            # May be in time-freq region: : (Batch, [Time, Channel, Freq])
            feats = transform(org_feats, train=False)
        else:
            feats = org_feats

        with torch.no_grad():
            enhanced, mask, ilens = model.enhance(feats)

        for idx, name in enumerate(names):
            # Assuming mask, feats : [Batch, Time, Channel. Freq]
            #          enhanced    : [Batch, Time, Freq]
            enh = enhanced[idx][: ilens[idx]]
            mas = mask[idx][: ilens[idx]]
            feat = feats[idx]

            # Plot spectrogram
            if args.image_dir is not None and num_images < args.num_images:
                import matplotlib.pyplot as plt

                num_images += 1
                ref_ch = 0

                plt.figure(figsize=(20, 10))
                plt.subplot(4, 1, 1)
                plt.title("Mask [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    mas[:, ref_ch].T,
                    fs=args.fs,
                    mode="linear",
                    frame_shift=frame_shift,
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 2)
                plt.title("Noisy speech [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    feat[:, ref_ch].T,
                    fs=args.fs,
                    mode="db",
                    frame_shift=frame_shift,
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 3)
                plt.title("Masked speech [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    (feat[:, ref_ch] * mas[:, ref_ch]).T,
                    frame_shift=frame_shift,
                    fs=args.fs,
                    mode="db",
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 4)
                plt.title("Enhanced speech")
                plot_spectrogram(
                    plt, enh.T, fs=args.fs, mode="db", frame_shift=frame_shift
                )

                plt.savefig(os.path.join(args.image_dir, name + ".png"))
                plt.clf()

            # Write enhanced wave files
            if enh_writer is not None:
                if istft is not None:
                    enh = istft(enh)
                else:
                    enh = enh

                if args.keep_length:
                    if len(org_feats[idx]) < len(enh):
                        # Truncate the frames added by stft padding
                        enh = enh[: len(org_feats[idx])]
                    elif len(org_feats) > len(enh):
                        padwidth = [(0, (len(org_feats[idx]) - len(enh)))] + [
                            (0, 0)
                        ] * (enh.ndim - 1)
                        enh = np.pad(enh, padwidth, mode="constant")

                if args.enh_filetype in ("sound", "sound.hdf5"):
                    enh_writer[name] = (args.fs, enh)
                else:
                    # Hint: To dump stft_signal, mask or etc,
                    # enh_filetype='hdf5' might be convenient.
                    enh_writer[name] = enh

            if num_images >= args.num_images and enh_writer is None:
                logging.info("Breaking the process.")
                break


def ctc_align(args):
    """CTC forced alignments with the given args.

    Args:
        args (namespace): The program arguments.
    """

    def add_alignment_to_json(js, alignment, char_list):
        """Add N-best results to json.

        Args:
            js (dict[str, Any]): Groundtruth utterance dict.
            alignment (list[int]): List of alignment.
            char_list (list[str]): List of characters.

        Returns:
            dict[str, Any]: N-best results added utterance dict.

        """
        # copy old json info
        new_js = dict()
        new_js["ctc_alignment"] = []

        alignment_tokens = []
        for idx, a in enumerate(alignment):
            alignment_tokens.append(char_list[a])
        alignment_tokens = " ".join(alignment_tokens)

        new_js["ctc_alignment"] = alignment_tokens

        return new_js

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()

    # read json data
    with open(args.align_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}
    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) aligning " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat, label = load_inputs_and_targets(batch)
                feat = feat[0]
                label = label[0]
                enc = model.encode(torch.as_tensor(feat).to(device)).unsqueeze(0)
                alignment = model.ctc.forced_align(enc, label)
                new_js[name] = add_alignment_to_json(
                    js[name], alignment, train_args.char_list
                )
    else:
        raise NotImplementedError("Align_batch is not implemented.")

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
