# Note: export paths if needed
# export PYTHONPATH="/home/gs534/rds/hpc-work/work/espnet-mm:$PYTHONPATH"
# export PATH="/home/gs534/rds/hpc-work/work/espnet-mm/tools/venv/bin:$PATH"
# export PYTHONPATH="/home/gs534/rds/hpc-work/work/espnet-mm/espnet/nets/pytorch_backend/lm:$PYTHONPATH"
. ./path.sh || exit 1;

train_config=conf/train_slu_tcpgen_gcn.yaml
backend=pytorch
expname=slurp_conformer_KA2G
preprocess_config=conf/specaug_tm.yaml

# lexicon_path=/home/dawna/gs534/espnet-KG/egs/ami/ami_lextree/data/local/wordlm_train/rarewords.txt
expdir=/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/slurp/exp/${expname}
debugmode=1
bpemode=unigram
nbpe=600
suffix='suffix'
KB='_KBf30onto'
context='_slufull'
dict=/home/gs534/rds/hpc-work/work/slurp/data/lang_char/train_960_${bpemode}${nbpe}${suffix}_units.txt
# dict=data/lang_1char/ihm_train_units.txt
mkdir -p ${expdir}

python ../../../espnet/bin/asr_train.py \
    --config ${train_config} \
    --ngpu 1 \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --tensorboard-dir tensorboard/${expname} \
    --preprocess-conf ${preprocess_config} \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --debugdir ${expdir} \
    --minibatches 0 \
    --verbose 0 \
    --seed 1 \
    --train-json /home/gs534/rds/hpc-work/work/slurp/dump/train/deltafalse/data_${bpemode}${nbpe}${suffix}${KB}${context}.json \
    --valid-json /home/gs534/rds/hpc-work/work/slurp/dump/devel/deltafalse/data_${bpemode}${nbpe}${suffix}${KB}${context}.json \
    --resume /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/slurp/exp/slurp_conformer_sepslottcpgen20_10distractor_noproj_nomask_sche15_copysup_GCNslot3in_f30_maskedless_share/results/resume.ep.20 \
    # --resume ${expdir}/results/snapshot.ep.27 \
    # --resume /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/slurp/exp/slurp_conformer_sepslottcpgen20_10distractor_noproj_nomask_sche15_copysup_f30/results/resume.ep.16
    # --suboperation concatshift \
    # --wordlevel true \
    # --beforeLSTM \
    # --charbased \
    # --stack-input \
    # --train-json dump/ihm_train/deltafalse/data_bpe_debug.json \
    # --valid-json dump/ihm_dev/deltafalse/data_bpe_debug.json \
