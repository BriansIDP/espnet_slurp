#!/usr/bin/env python3

"""Transducer loss module."""

import torch


class TransLoss(torch.nn.Module):
    """Transducer loss module.

    Args:
        trans_type (str): type of transducer implementation to calculate loss.
        blank_id (int): blank symbol id
    """

    def __init__(self, trans_type, blank_id, useKB=False):
        """Construct an TransLoss object."""
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if trans_type == "warp-transducer":
            from warprnnt_pytorch import RNNTLoss

            self.trans_loss = RNNTLoss(blank=blank_id)
        elif trans_type == "warp-rnnt":
            if device.type == "cuda":
                try:
                    from warp_rnnt import rnnt_loss

                    self.trans_loss = rnnt_loss
                except ImportError:
                    raise ImportError(
                        "warp-rnnt is not installed. Please re-setup"
                        " espnet or use 'warp-transducer'"
                    )
            else:
                raise ValueError("warp-rnnt is not supported in CPU mode")

        self.trans_type = trans_type
        self.blank_id = blank_id
        self.useKB = useKB

    def forward(self, pred_pad, target, pred_len, target_len, ptr_dist=None, p_gen=None):
        """Compute path-aware regularization transducer loss.

        Args:
            pred_pad (torch.Tensor): Batch of predicted sequences
                (batch, maxlen_in, maxlen_out+1, odim)
            target (torch.Tensor): Batch of target sequences (batch, maxlen_out)
            pred_len (torch.Tensor): batch of lengths of predicted sequences (batch)
            target_len (torch.tensor): batch of lengths of target sequences (batch)

        Returns:
            loss (torch.Tensor): transducer loss

        """
        dtype = pred_pad.dtype
        if dtype != torch.float32:
            # warp-transducer and warp-rnnt only support float32
            pred_pad = pred_pad.to(dtype=torch.float32)

        if self.trans_type == "warp-rnnt":
            if self.useKB and ptr_dist is not None and p_gen is not None:
                model_dist = torch.softmax(pred_pad, dim=-1)
                # only use pointer when null probability is below a certain value
                # p_gen.masked_fill_(model_dist[:,:,:,0:1] >= 0.8, 0)
                p_not_null = 1.0 - model_dist[:,:,:,0:1]
                ptr_dist_fact = ptr_dist[:,:,:,1:] * p_not_null
                ptr_gen_complement = (ptr_dist[:,:,:,-1:]) * p_gen
                p_partial = ptr_dist_fact[:,:,:,:-1] * p_gen + model_dist[:,:,:,1:] * (1 - p_gen + ptr_gen_complement)
                p_final = torch.cat([model_dist[:,:,:,0:1], p_partial], dim=-1)
                log_probs = torch.log(p_final)
            else:
                log_probs = torch.log_softmax(pred_pad, dim=-1)

            loss = self.trans_loss(
                log_probs,
                target,
                pred_len,
                target_len,
                reduction="mean",
                blank=self.blank_id,
                gather=True,
            )
        else:
            loss = self.trans_loss(pred_pad, target, pred_len, target_len)
        loss = loss.to(dtype=dtype)

        return loss
