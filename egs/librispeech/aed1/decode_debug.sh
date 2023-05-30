source ~/.bashrc
export PYTHONPATH="/home/mifs/gs534/Documents/Project/exp/espnet:$PYTHONPATH"
export PYTHONPATH="/home/mifs/gs534/Documents/Project/exp/espnet/espnet/nets/pytorch_backend/lm:$PYTHONPATH"
. ./path.sh || exit 1;
echo "Start Decoding"
echo "pythonpath = $PYTHONPATH"
nj=16
recog_set="test_clean_100"
# recog_set="train_100"
use_lm=false
use_wordlm=false
dumpdir=/data/mifs_scratch/gs534/librispeech/dump/
do_delta=false
backend=pytorch

# Task specific
# lmexpdir=/home/gs534/rds/hpc-work/work/AMI/exp/external_rnnlm
tag="debug"
expdir=/data/mifs_scratch/gs534/librispeech/exp/Librispeech100_cfmaed_KB500_nodrop_gcnii6l
recog_model=model.loss.best
decode_config=conf/decode.yaml
bpemode=unigram
nbpe=600
suffix='suffix'
dict=/data/mifs_scratch/gs534/librispeech/data_100/lang_char/train_960_${bpemode}${nbpe}${suffix}_units.txt

pids=() # initialize pids
for rtask in ${recog_set}; do
(
    decode_dir=decode_${rtask}_${tag}

    if [ ${use_lm} = true ]; then
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best --external true"
        fi
    else
        echo "No language model is involved."
        recog_opts=""
    fi

    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_unigram600RNNT_KBfull.json
    mkdir ${expdir}/${decode_dir}

    #### use CPU for decoding
    ngpu=0

    asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json ${feat_recog_dir}/split128utt/data_unigram600RNNT_KBfull.39.json \
        --result-label ${expdir}/${decode_dir}/data.1.json \
        --model ${expdir}/results/${recog_model}  \
        ${recog_opts} \
        # --external \
        # --suboperation concatshift \
        # --crossutt memorynet \
        # --cont \
        # --crossutt \
)
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
