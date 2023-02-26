### Note: export paths if needed
# export PYTHONPATH="/home/gs534/rds/hpc-work/work/espnet-mm:$PYTHONPATH"
# export PATH="/home/gs534/rds/hpc-work/work/espnet-mm/tools/venv/bin:$PATH"
# export PYTHONPATH="/home/gs534/rds/hpc-work/work/espnet-mm/espnet/nets/pytorch_backend/lm:$PYTHONPATH"
. ./path.sh || exit 1;

### Need to modify expdir and dict to the correct path, and also modify train-json and valid-json
train_config=conf/train_slu_tcpgen_gcn.yaml
backend=pytorch
expname=slurp_conformer_KA2G
preprocess_config=conf/specaug_tm.yaml
expdir=exp/${expname}
debugmode=1
bpemode=unigram
nbpe=600
suffix='suffix'
KB='_KBf30onto'
context='_slufull'
# The word piece dictionary from LibriSpeech 960, or you can change to your own dictionary with your own model
dict=data/lang_char/train_960_${bpemode}${nbpe}${suffix}_units.txt
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
    --train-json dump/train/deltafalse/data_${bpemode}${nbpe}${suffix}${KB}${context}.json \
    --valid-json dump/devel/deltafalse/data_${bpemode}${nbpe}${suffix}${KB}${context}.json \
    # --resume /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/slurp/exp/slurp_conformer_sepslottcpgen20_10distractor_noproj_nomask_sche15_copysup_GCNslot3in_f30_maskedless_share/results/resume.ep.20 \
