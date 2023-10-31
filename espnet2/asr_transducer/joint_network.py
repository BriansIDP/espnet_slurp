"""Transducer joint network implementation."""

import torch

from espnet2.asr_transducer.activation import get_activation


class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        output_size: Output size.
        encoder_size: Encoder output size.
        decoder_size: Decoder output size.
        joint_space_size: Joint space size.
        joint_act_type: Type of activation for joint network.
        **activation_parameters: Parameters for the activation function.

    """

    def __init__(
        self,
        output_size: int,
        encoder_size: int,
        decoder_size: int,
        joint_space_size: int = 256,
        joint_activation_type: str = "tanh",
        deepbiasing: bool = False,
        biasingsize: int = 256,
        biasing: bool = False,
        lin_dec_bias: bool = True,
        **activation_parameters,
    ) -> None:
        """Construct a JointNetwork object."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(
            decoder_size, joint_space_size, bias=lin_dec_bias
        )

        self.lin_out = torch.nn.Linear(joint_space_size, output_size)

        self.joint_activation = get_activation(
            joint_activation_type, **activation_parameters
        )
        # biasing
        self.joint_space_size = joint_space_size
        self.deepbiasing = deepbiasing
        self.biasing = biasing
        if biasing and deepbiasing:
            self.lin_biasing = torch.nn.Linear(biasingsize, joint_space_size)

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        no_projection: bool = False,
        hptr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Joint computation of encoder and decoder hidden state sequences.

        Args:
            enc_out: Expanded encoder output state sequences.
                         (B, T, s_range, D_enc) or (B, T, 1, D_enc)
            dec_out: Expanded decoder output state sequences.
                         (B, T, s_range, D_dec) or (B, 1, U, D_dec)

        Returns:
            joint_out: Joint output state sequences.
                           (B, T, U, D_out) or (B, T, s_range, D_out)

        """
        if self.deepbiasing and hptr is not None:
            joint_out = self.joint_activation(
                self.lin_enc(enc_out) + self.lin_dec(dec_out) + self.lin_biasing(hptr)
            )
        elif no_projection:
            joint_out = self.joint_activation(enc_out + dec_out)
        else:
            joint_out = self.joint_activation(
                self.lin_enc(enc_out) + self.lin_dec(dec_out)
            )

        if self.biasing:
            return self.lin_out(joint_out), joint_out
        else:
            return self.lin_out(joint_out)
