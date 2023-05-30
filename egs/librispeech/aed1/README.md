# This is the Conformer AED implementation with TCPGen

This directory contains sample implementations of training and evaluation pipelines for the Conformer AED model with tree-constrained pointer generator (TCPGen) for contextual biasing, as described in the paper: [Tree-Constrained Pointer Generator for End-to-End Contextual Speech Recognition](https://ieeexplore.ieee.org/abstract/document/9687915)

## Setup
1. Install ESPNet as instructed [here](https://espnet.github.io/espnet/installation.html)
2. Install `tensorboardX`

## Feature Extraction
Using the recipe in [../asr1](../asr1/run.sh) to extract 80-dim filterbank features following stage 0 to 2. Note that this version of TCPGen only supports suffix-based word piece, so please add `--treat_whitespace_as_suffix=true` when using `spm_train` in stage 2. 

We also provide our modified recipe for feature generation `run.sh` here that was used in the original paper, which does not include speed perturbation, using 600 suffix-based unigram WordPieces and is only for train-clean-100. For the full training set, we uses 120 as the frequency threshold. You need to specify your data path to the full training set and run the recipe again from beginning.

## Biasing list preparation
Prepare biasing list for training and evaluation separately. 

### Training
The following example shows how to prepare and train a Conformer AED system with TCPGen and GNN encodings on the full 960-hour LibriSpeech training set.

The biasing list for training is extracted by counting words appeared in training, and selecting ones with frequency less than 120, and save them as a list of words in [data/Biasing/Librispeech_blists/rareword_f120.txt](data/Biasing/Librispeech_blists/rareword_f120.txt). To avoid searching this list over and over again during training and tokenise words into WordPieces, we add biasing words appeared in each utterance into training JSON file:

1. Convert the list into WordPiece format, e.g. [data/Biasing/Librispeech_unigram600suffix/rareword_f120.txt](data/Biasing/Librispeech_unigram600suffix/rareword_f120.txt)
2. Make a mapping between words and WordPiece for all words in the biasing list: e.g. [data/Biasing/bpe_dict_word_unigram600suffix_train960.txt](data/Biasing/bpe_dict_word_unigram600suffix_train960.txt)
3. Use the script [dump/train/deltafalse/get_biasinglist.py](dump/train_100/deltafalse/get_biasinglist.py) to generate [data_unigram600suffix_KB120.json](train/deltafalse/data_unigram600suffix_KB120.json) for training, and also repeat the above for validation set. 

Of course you can get data_unigram600suffix_KBf120.json via other methods/script designs, as long as it generates in the same format so that the code can read it.

### Inference
Repeat above procedure, but with a different biasing list provided by [Meta research](https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias). We include that biasing list at [data/Biasing/Librispeech_blists/all_rare_words.txt](data/Biasing/Librispeech_blists/all_rare_words.txt). Example test file in [dump/test_clean/deltafalse/data_unigram600suffix_KBfull.json](dump/test_clean/deltafalse/data_unigram600suffix_KBfull.json)

Note that the same training and inference example JSON files are provided for experiments with train-clean-100. See [data/train_100](data/train_100) and [data/test_clean_100](data/test_clean_100) for examples.


## Training
The training script is `train.sh`. Please replace the `PYTHONPATH`, `$dict` and train and valid json files with your own paths. 

In the conf/train_KB_conformer.yaml, you need to specify:
1. `meetingpath`: The path to the wordpiece level biasing list: e.g. [data/Biasing/Librispeech_unigram600suffix/rareword_f120.txt](data/Biasing/Librispeech_unigram600suffix/rareword_f120.txt)
2. `dictfile`: The path to the list of all training set words in WordPiece format

Expected validation set accuracy for train-clean-100 with unigram 600 suffix WordPieces is around 91% with TCPGen, for 80 epochs. 

## Decoding
**Important!!** Before decoding, you need to modify the `model.json` file the following entries:
1. Change `"DBdrop"` to 0.0. During training it was set to 0.3 but we don't drop any biasing words during testing.
2. Change `"KBmaxlen"` to whatever size you want to test, e.g. 1000 which is used in the paper. (Note `"KBminlen"` is not used in the code)
3. Change the `"meetingpath"` to the 200k-word biasing list in WordPiece format, e.g. [data/Biasing/Librispeech_unigram600suffix/all_rare_words_train960.txt](data/Biasing/Librispeech_unigram600suffix/all_rare_words_train960.txt)

The SGE submission code for decoding is `submit_decode.sh`, which submit single jobs of `decode_bpe.sh`. Please adapt this script to your local resources. Replace all paths in `submit_decode.sh` with your local path.

## Scoring
Please install the NIST SCTK scoring tool for WER here if you haven't [https://github.com/usnistgov/SCTK/blob/master/README.md](https://github.com/usnistgov/SCTK/blob/master/README.md). Note ESPNet may have installed it automatically. This will generate a file named `results.wrd.txt` under the decoding directory.

In addition to normal WER scoring, to score rare word error rates (R-WER):
`cd error_analysis`
`python get_error_word_count.py <path_to_results.wrd.txt>`

This will give you the rare word error rate.

Expected WER for test-clean:

|                     |          WER |      R-WER |
|:-------------------:|-------------:|-----------:|
| test-clean baseline          |       0.0371 |      0.1320|
| test-clean TCPGen          |       0.0334 |      0.0840|
| test-clean TCPGen + GNN          |       0.0259 |      0.0580|

|                     |          WER |      R-WER |
|:-------------------:|-------------:|-----------:|
| test-other baseline         |       0.0936 |      0.2950|
| test-other TCPGen         |       0.0843 |      0.2130|
| test-other TCPGen + GNN         |       0.0700 |      0.1550|
