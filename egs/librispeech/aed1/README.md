# This is the Conformer AED implementation with TCPGen

This directory contains sample implementations of training and evaluation pipelines for the Conformer AED model with tree-constrained pointer generator (TCPGen) for contextual biasing, as described in the paper: [Tree-Constrained Pointer Generator for End-to-End Contextual Speech Recognition](https://ieeexplore.ieee.org/abstract/document/9687915)

## Setup
1. Install ESPNet as instructed [here](https://espnet.github.io/espnet/installation.html)
2. Install `tensorboardX`

## Feature Extraction
Using the recipe in [../asr1](../asr1/run.sh) to extract 80-dim filterbank features following stage 0 to 2. Note that this version of TCPGen only supports suffix-based word piece, so please add `--treat_whitespace_as_suffix=true` when using `spm_train` in stage 2. 

We also provide our modified recipe for feature generation `run.sh` here that was used in the original paper, which does not include speed perturbation, and is only for train-clean-100. 

## Biasing list preparation
Prepare biasing list for training and evaluation separately. 

### Training
The biasing list for training is extracted by counting words appeared in training, and selecting ones with frequency less than 15, and save them as a list of words in [data/Biasing/Librispeech_blists/rareword_f15.txt](data/Biasing/Librispeech_blists/rareword_f15.txt)

