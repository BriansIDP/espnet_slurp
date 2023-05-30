export PYTHONPATH="/home/mifs/gs534/Documents/Project/exp/espnet:$PYTHONPATH"
export PYTHONPATH="/home/mifs/gs534/Documents/Project/exp/espnet/espnet/nets/pytorch_backend/lm:$PYTHONPATH"
. ./path.sh || exit 1;
echo "Start Decoding"
echo "pythonpath = $PYTHONPATH"
nj=16
# cmd="queue.pl -l qp=low,osrel=*,not_host='air120|air113|air112' -P black-svr"
cmd="run.pl"
recog_set="test_clean_100"
# recog_set="train_100"
use_lm=false
use_wordlm=false
dumpdir=/data/mifs_scratch/gs534/librispeech/dump
do_delta=false
backend=pytorch

# Task specific
lmexpdir=/home/gs534/rds/hpc-work/work/AMI/exp/external_rnnlm
tag="no_lm_b30"
# expdir=exp/ami_train_transformer_concat
expdir=/data/mifs_scratch/gs534/librispeech/exp/Librispeech100_cfmaed_KB500_nodrop_gcnii6l
recog_model=model.loss.best
decode_config=conf/decode.yaml
bpemode=unigram
nbpe=600
suffix='suffix'
# Need to replace it with your local one
dict=/data/mifs_scratch/gs534/librispeech/data_100/lang_char/train_960_${bpemode}${nbpe}${suffix}_units.txt

pids=() # initialize pids
for rtask in ${recog_set}; do
(
    decode_dir=decode_${rtask}_${tag}
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}${suffix}_KBfull.json
    ${cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log ./decode_bpe.sh JOB ${rtask} \
        ${decode_dir} \
        ${feat_recog_dir} \
        ${expdir} \
        ${nj} \
        ${use_lm} \
        ${use_wordlm} \
        ${decode_config} \
        ${recog_model} \
        ${dict} \
        ${lmexpdir} \
        ${bpemode} \
        ${nbpe} \
        ${fixedemb} \
) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
