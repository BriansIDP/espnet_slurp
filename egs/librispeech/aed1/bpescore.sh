. ./path.sh || exit 1;

set="test_clean_100"
# set="train_100"
lm="no_lm_b30"
# lm="debug"
# lm='wordpiece_lm_b30_0.25_ilme0.25'
nbpe=600RNNT
# nbpe=50000
bpemode=unigram
bpemodel=/data/mifs_scratch/gs534/librispeech/data_100/lang_char/train_960_${bpemode}${nbpe}
# lm="external_wordlm_0.1"
# expdir=/home/gs534/rds/hpc-work/work/Librispeech/exp_100_mm/librispeech_train_conformer_baseline
expdir=/data/mifs_scratch/gs534/librispeech/exp/Librispeech100_cfmrnnt_KB500_nodrop_gcn5l_tied_2heads_0.1_0.9
# expdir=/home/gs534/rds/hpc-work/work/AMI/rnnt_exp/ami_train_rnnt_KB100_drop_ooKB
# expdir=/home/gs534/rds/hpc-work/work/AMI/rnnt_exp/ami_train_rnnt_KB100_KBin_0.1drop
decode_dir=decode_${set}_${lm}
dict=/data/mifs_scratch/gs534/librispeech/data_100/lang_char/train_960_${bpemode}${nbpe}_units.txt
score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
