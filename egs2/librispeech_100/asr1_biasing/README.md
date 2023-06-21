# TCPGen in RNN-T
## TCPGen

## asr_train_conformer_transducer_tcpgen500_deep_sche30_suffix

- ASR Config: [conf/train_rnnt.yaml](conf/train_rnnt.yaml)
- Params: 27.13M

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|52576|94.4|4.1|0.5|0.7|5.2|50.4|

## asr_train_conformer_transducer_tcpgen500_deep_sche30_suffix

- ASR Config: [conf/tuning/train_rnnt_std_tcpgen.yaml](conf/tuning/train_rnnt_std_tcpgen.yaml)
- Params: 26.99M

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|52576|94.4|4.5|0.5|0.7|5.7|54.3|
