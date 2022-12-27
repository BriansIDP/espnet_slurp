#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

asr_config=conf/train_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 600 \
    --token_type word\
    --audio_format flac\
    --feats_type fbank_pitch\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 8 \
    --inference_asr_model valid.acc.ave_10best.pth\
    --stage 1 \
    --stop_stage 6 \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
