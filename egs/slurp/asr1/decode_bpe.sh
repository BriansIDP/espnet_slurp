. ./path.sh || exit 1;
export PYTHONPATH="/home/gs534/rds/hpc-work/work/espnet-mm:$PYTHONPATH"
export PATH="/home/gs534/rds/hpc-work/work/espnet/tools/venv/bin:$PATH"
export PYTHONPATH="/home/gs534/rds/hpc-work/work/espnet/espnet/nets/pytorch_backend/lm:$PYTHONPATH"
echo "Start Decoding"
echo "pythonpath = $PYTHONPATH"
recog_set="ihm_dev ihm_eval"
do_delta=false
backend=pytorch
JOB=$1
rtask=$2
decode_dir=$3
feat_recog_dir=$4
expdir=$5
nj=$6
use_lm=$7
use_wordlm=$8
# Task specific
decode_config=$9
recog_model=${10}
dict=${11}
lmexpdir=${12}
bpemode=${13}
nbpe=${14}
fixedemb=${15}

echo ${recog_model}
if [ ${use_lm} = true ]; then
    if [ ${use_wordlm} = true ]; then
        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        lang_opts='--cont --external'
    else
        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best --external true"
        lang_opts=''
    fi
else
    echo "No language model is involved."
    recog_opts=""
fi

# split data
# splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

#### use CPU for decoding
ngpu=0

asr_recog.py \
    --config ${decode_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}_empty.${JOB}.json \
    --result-label ${expdir}/${decode_dir}/data.${JOB}.json \
    --model ${expdir}/results/${recog_model}  \
    ${recog_opts} \
    ${lang_opts} \
