from __future__ import print_function
import torch.nn as nn
import torch
from torch import cat, randn_like
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, nclasses=1, rnndrop=0.5, dropout=0.5, tie_weights=False, reset=0):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=rnndrop)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp and rnn_type != "LSTMP":
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        # self.TPbottleneck = nn.Linear(nhid, 128)
        self.TPHead = nn.Linear(nhid, nclasses)
        self.ff1 = nn.Linear(ninp, ninp)
        self.ff2 = nn.Linear(ninp, ninp)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers
        self.nclasses = nclasses
        self.reset = reset
        self.mode = 'train'
        self.grads = {}

    def set_mode(self, m):
        self.mode = m

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.TPHead.weight.data.uniform_(-initrange, initrange)
        self.TPHead.bias.data.zero_()
        self.ff1.weight.data.uniform_(-initrange, initrange)
        self.ff2.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, temperature=1, seedwords=None):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(self.drop(output)) / temperature
        # TPdecoded = self.TPbottleneck(self.drop(output))
        TPdecoded = self.TPHead(self.drop(output))
        return decoded, hidden, TPdecoded

    def forward_reset(self, input, hidden, temperature=1, eosidx=1):
        emb = self.drop(self.encoder(input))
        output_list = []
        for i in range(emb.size(0)):
            resethidden = self.resetsent(hidden, input[i,:], eosidx)
            each_output, hidden = self.rnn(emb[i,:,:].view(1,emb.size(1),-1), resethidden)
            output_list.append(each_output)
        output = cat(output_list, 0)
        output = self.drop(output)
        decoded = self.decoder(output) / temperature
        TPdecoded = self.TPHead(output)
        return decoded, hidden, TPdecoded

    def fusion(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output, hidden

    def forward_firstorder(self, targets, inputs, rmask):
        candidates = []
        rare_targets = []
        for i, mask in enumerate(rmask):  
            if mask.item() == 1:
                candidates.append(inputs[i:i+1])
                rare_targets.append(targets[i:i+1])
        candidates = torch.cat(candidates, dim=0)
        targets = torch.cat(rare_targets)
        mapped_embs = self.encoder(candidates)
        # mapped_embs = torch.relu(self.ff1(mapped_embs))
        # mapped_embs = self.ff2(mapped_embs)
        targets = self.encoder(targets)
        mse = ((mapped_embs - targets.unsqueeze(1))**2).mean()
        return mse

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def resetsent(self, hidden, input, eosidx):
        outputcell = hidden[0]
        memorycell = hidden[1]
        mask = input != eosidx
        expandedmask = mask.unsqueeze(-1).expand_as(outputcell)
        expandedmask = expandedmask.float()
        return (outputcell*expandedmask, memorycell*expandedmask)

class RNNKBModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, rnndrop=0.5, dropout=0.5,
                 tie_weights=False, reset=0, seedword=False, nattn=0):
        super(RNNKBModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=rnndrop)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp and rnn_type != "LSTMP":
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        if seedword:
            self.nattn = nattn
            self.Kproj = nn.Linear(ninp, self.nattn)
            self.Qproj = nn.Linear(nhid, self.nattn)
            self.seeddrop = nn.Dropout(0.1)
            self.TPHead = nn.Linear(nhid+self.nattn, nhid)
        else:
            self.TPHead = nn.Linear(nhid, ntoken)

        self.seedword = seedword
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers
        self.reset = reset
        self.mode = 'train'
        self.grads = {}
        self.init_weights()

    def set_mode(self, m):
        self.mode = m

    def load_from(self, model_init):
        model_init = torch.load(model_init, map_location=lambda storage, loc: storage)
        model_init = model_init.state_dict() if not isinstance(model_init, dict) else model_init
        own_state = self.state_dict()
        for name, param in model_init.items():
            if name in own_state and 'TPHead' not in name:
                own_state[name].copy_(param.data)
        for name, param in self.named_parameters():
            if name in model_init and 'TPHead' not in name:
                param.requires_grad = False

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.TPHead.weight.data.uniform_(-initrange, initrange)
        self.TPHead.bias.data.zero_()
        if self.seedword:
            self.Kproj.weight.data.uniform_(-initrange, initrange)
            self.Kproj.bias.data.zero_()
            self.Qproj.weight.data.uniform_(-initrange, initrange)
            self.Qproj.bias.data.zero_()

    def forward(self, input, hidden, temperature=1, seedwords=None):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(self.drop(output)) / temperature
        if seedwords is not None and self.seedword:
            seedword_emb = self.drop(self.encoder(seedwords))
            query = self.drop(self.Qproj(output))
            keys = self.drop(self.Kproj(seedword_emb))
            attn_weights = torch.einsum('ijk,ijsk->ijs', query, keys)
            attn_weights = attn_weights / query.size(-1) ** 0.5
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.einsum('ijs,ijsk->ijk', attn_weights, keys)
            output = torch.cat([self.drop(output), attn_output], dim=-1)
            # TPdecoded = self.TPHead(output)
            TPdecoded = self.decoder(self.drop(self.TPHead(output)))
        else:
            TPdecoded = self.TPHead(self.drop(output))
        return decoded, hidden, TPdecoded

    def forward_reset(self, input, hidden, temperature=1, eosidx=1, seedwords=None):
        emb = self.drop(self.encoder(input))
        output_list = []
        for i in range(emb.size(0)):
            resethidden = self.resetsent(hidden, input[i,:], eosidx)
            each_output, hidden = self.rnn(emb[i,:,:].view(1,emb.size(1),-1), resethidden)
            output_list.append(each_output)
        output = cat(output_list, 0)
        output = self.drop(output)
        decoded = self.decoder(output) / temperature
        if seedwords is not None and self.seedword:
            seedword_emb = self.drop(self.encoder(seedwords))
            query = self.seeddrop(self.Qproj(output))
            keys = self.seeddrop(self.Kproj(seedword_emb))
            attn_weights = torch.einsum('ijk,ijsk->ijs', query, keys)
            attn_output = torch.einsum('ijs,ijsk->ijk', attn_weights, keys)
            output = torch.cat([output, attn_output], dim=-1)
            TPdecoded = self.decoder(self.TPHead(output))
        else:
            TPdecoded = self.TPHead(self.drop(output))
        return decoded, hidden, TPdecoded

    def fusion(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def resetsent(self, hidden, input, eosidx):
        outputcell = hidden[0]
        memorycell = hidden[1]
        mask = input != eosidx
        expandedmask = mask.unsqueeze(-1).expand_as(outputcell)
        expandedmask = expandedmask.float()
        return (outputcell*expandedmask, memorycell*expandedmask)
