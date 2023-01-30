# This is the Conformer AED implementation with TCPGen

## Setup
1. Install ESPNet as instructed [here](https://espnet.github.io/espnet/installation.html)
2. Install `tensorboardX`

## Feature Extraction
Using the recipe in [../asr1](../asr1/run.sh) to extract 80-dim filterbank features following stage 0 to 2. Note that this version of TCPGen only supports suffix-based word piece, so please add `--treat_whitespace_as_suffix=true` when using `spm_train` in stage 2. 

We also provide our modified recipe for feature generation `run.sh` here, which does not include speed perturbation, and is only for train-clean-100. 

## Biasing list preparation
Prepare biasing list for training and evaluation.

