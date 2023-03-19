export PYTHONPATH="/scratch/OpenSource/espnet:$PYTHONPATH"
export PYTHONPATH="/scratch/OpenSource/espnet/espnet/nets/pytorch_backend/lm:$PYTHONPATH"
. ./path.sh || exit 1;

train_config=conf/train_TCPGen_100.yaml
backend=pytorch
expname=Librispeech_AED_TCPGen_GCNII2L
preprocess_config=conf/specaug_tm.yaml

# lexicon_path=/home/dawna/gs534/espnet-KG/egs/ami/ami_lextree/data/local/wordlm_train/rarewords.txt
expdir=exp/${expname}
debugmode=1
bpemode=unigram
nbpe=600
suffix='suffix'
KB='_KB'
context=''
dict=/data/mifs_scratch/gs534/librispeech/data_100/lang_char/train_960_${bpemode}${nbpe}${suffix}_units.txt
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
    --train-json /data/mifs_scratch/gs534/librispeech/dump/train_100/deltafalse/data_${bpemode}${nbpe}${suffix}${KB}${context}.json \
    --valid-json /data/mifs_scratch/gs534/librispeech/dump/dev_100/deltafalse/data_${bpemode}${nbpe}${suffix}${KB}${context}.json \
