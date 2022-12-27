from copy import deepcopy
import chainer
import torch
import logging


def check_early_stop(trainer, epochs):
    """Checks an early stopping trigger and warns the user if it's the case

    :param trainer: The trainer used for training
    :param epochs: The maximum number of epochs
    """
    end_epoch = trainer.updater.get_iterator("main").epoch
    if end_epoch < (epochs - 1):
        logging.warning(
            "Hit early stop at epoch "
            + str(end_epoch)
            + "\nYou can change the patience or set it to 0 to run all epochs"
        )


def set_early_stop(trainer, args, is_lm=False):
    """Sets the early stop trigger given the program arguments

    :param trainer: The trainer used for training
    :param args: The program arguments
    :param is_lm: If the trainer is for a LM (epoch instead of epochs)
    """
    patience = args.patience
    criterion = args.early_stop_criterion
    epochs = args.epoch if is_lm else args.epochs
    mode = "max" if "acc" in criterion else "min"
    if patience > 0:
        trainer.stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(
            monitor=criterion,
            mode=mode,
            patients=patience,
            max_trigger=(epochs, "epoch"),
        )


class EMA(torch.nn.Module):

    def __init__(self, model: torch.nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)
        # Share the same reporter
        self.shadow.reporter = self.model.reporter

        for param in self.shadow.parameters():
            param.detach_()
    
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except torch.nn.modules.module.ModuleAttributeError:
            return getattr(self.model, item)

    @torch.no_grad()
    def update_ema(self):
        assert self.training == True, "EMA update should only be called during training."

        model_params = dict(self.model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())
        assert model_params.keys() == shadow_params.keys()
        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = dict(self.model.named_buffers())
        shadow_buffers = dict(self.shadow.named_buffers())
        assert model_buffers.keys() == shadow_buffers.keys()
        for name, buffer in model_buffers.items():
            # EMA for moving mean/var as well
            if 'running_mean' in name or 'running_var' in name:
                shadow_buffers[name].sub_((1 - self.decay) * (shadow_buffers[name] - buffer))
            else:
                shadow_buffers[name].copy_(buffer)

    def forward(self, *inputs) -> torch.Tensor:
        """Forward original params by default."""
        return self.model(*inputs)
