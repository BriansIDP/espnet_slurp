import json
  
nbpe = 600
with open('data_unigram{}suffix_train960.json'.format(nbpe)) as fin:
    data = json.load(fin)

dictionary = {}
# Change it to your path to bpe_dict_word_unigram${nbpe}suffix_train960.txt file
with open('/home/dawna/gs534/espnet-debug/egs/librispeech/asr1/data/KBs/bpe_dict_word_unigram{}suffix.txt'.format(nbpe)) as fin:
    for line in fin:
        word = line.split()[0]
        bpes = tuple(line.split()[1:])
        dictionary[bpes] = word

# Change this to your path to Librispeech_unigram${nbpe}suffix/
with open('../../../data/Biasing/Librispeech_unigram{}suffix/rareword_f120.txt'.format(nbpe)) as fin:
    rarewords = {}
    for line in fin:
        bpes = tuple(line.split())
        word = dictionary[bpes]
        rarewords[word] = tuple(line.split())

newdata = {'utts':{}}
for utt, uttinfo in data['utts'].items():
    uttKB = []
    for word in uttinfo['output'][0]['text'].split():
        if word in rarewords and rarewords[word] not in uttKB:
            uttKB.append(rarewords[word])
    uttinfo['output'][0]['uttKB'] = uttKB
    newdata['utts'][utt] = uttinfo

with open('data_unigram{}suffix_KBf120.json'.format(nbpe), 'w') as fout:
    json.dump(newdata, fout, indent=4, ensure_ascii=False)
