import torch
from transformers import RobertaModel, RobertaTokenizer

from espnet.nets.pytorch_backend.nets_utils import to_device


class Roberta_encoder(torch.nn.Module):
    def __init__(self, model_name="roberta-base", pooling='mean', loadfrom=None):
        torch.nn.Module.__init__(self)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.pooling = pooling
        self.model = RobertaModel.from_pretrained(model_name)
        if loadfrom is not None:
            state_dict = torch.load(loadfrom, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(state_dict)
        # self.model.eval()

    def pooling_func(self, sequence_output, input_bounds):
        if self.pooling == 'max':
            input_bounds = input_bounds.transpose(1, 2)
            wordlevel_out = torch.einsum('ijk,imj->imjk', sequence_output, input_bounds)
            inf_mask = wordlevel_out.new_ones(wordlevel_out.size()) * -1e9 * (1 - input_bounds.unsqueeze(-1))
            wordlevel_out = (wordlevel_out + inf_mask).max(dim=2)[0] * (input_bounds.sum(dim=-1, keepdim=True) != 0)
        elif self.pooling == 'mean':
            wordlevel_out = torch.einsum('ijk,ijm->imk', sequence_output, input_bounds)
            wordlevel_out = wordlevel_out / torch.clamp(input_bounds.sum(dim=1), min=1).unsqueeze(-1)
        elif self.pooling == 'index':
            dummy = input_bounds.unsqueeze(-1).expand(input_bounds.size(0), input_bounds.size(1), sequence_output.size(2))
            wordlevel_out = torch.gather(sequence_output, 1, dummy) # sequence_output[torch.arange(input_bounds.size(0)), input_bounds]
        else:
            raise Exception("Unknown pooling function")
        return wordlevel_out

    def tokenize(self, input_text):
        tokenized_utts = []
        wordbound = []
        maxlen = 0
        for utt in input_text:
            tokenized_utt = self.tokenizer.tokenize(u'<s>')
            bound = [0] if self.pooling == 'index' else [1]
            for word in utt.split():
                word_tok = self.tokenizer.tokenize(' '+word)
                if self.pooling == 'index':
                    bound.append(len(tokenized_utt))
                else:
                    bound += [1] + [0] * (len(word_tok) - 1)
                tokenized_utt.extend(word_tok)
            if self.pooling == 'index':
                bound.append(len(tokenized_utt))
            else:
                bound.append(1)
            tokenized_utt.extend(self.tokenizer.tokenize(u'</s>'))
            if len(tokenized_utt) > maxlen:
                maxlen = len(tokenized_utt)
            tokenized_utts.append(tokenized_utt)
            wordbound.append(bound)
        return tokenized_utts, wordbound, maxlen

    def get_input_ids(self, tokenized_utts, maxlen):
        inputs = []
        input_masks = []
        for i, utt in enumerate(tokenized_utts):
            input_ids = self.tokenizer.convert_tokens_to_ids(utt)
            input_mask = [1] * len(input_ids)
            input_ids += [1] * (maxlen - len(input_ids))
            input_mask += [0] * (maxlen - len(input_mask))

            inputs.append(input_ids)
            input_masks.append(input_mask)
        inputs = to_device(self, torch.LongTensor(inputs))
        input_masks = torch.tensor(input_masks).to(inputs.device)
        return inputs, input_masks

    def get_wordbound_sum(self, wordbound, maxlen, input_masks):
        input_bounds = []
        for i, bound in enumerate(wordbound):
            input_bound = torch.cumsum(torch.tensor(bound), dim=0)-1
            bound_padding = input_bound.new_zeros((maxlen - input_bound.size(0)))
            input_bound = torch.cat([input_bound, bound_padding], dim=0)
            input_bounds.append(input_bound)
        input_bounds = torch.stack(input_bounds)
        input_bounds = torch.cat([input_bounds, input_bounds.new_ones(input_bounds.size(0), 1) * (maxlen - 1)], dim=-1)
        input_bounds = torch.nn.functional.one_hot(input_bounds)
        input_bounds = to_device(self, input_bounds[:,:-1])

        input_bounds = input_bounds * input_masks.unsqueeze(-1)
        wordmask = input_bounds.sum(dim=1) != 0
        wordmask[torch.arange(wordmask.size(0)), wordmask.sum(dim=1) - 1] = False
        return input_bounds.float(), wordmask

    def get_wordbound_idx(self, wordbound):
        maxlen = 0
        for bound in wordbound:
            if len(bound) > maxlen:
                maxlen = len(bound)
        input_bounds = []
        wordmask = []
        for bound in wordbound:
            wordmask.append([1] * len(bound) + [0] * (maxlen-len(bound)))
            input_bounds.append(bound + [bound[-1]] * (maxlen-len(bound)))
        input_bounds = to_device(self, torch.LongTensor(input_bounds))
        wordmask = to_device(self, torch.Tensor(wordmask))
        return input_bounds, wordmask

    def forward(self, input_text, training=False):
        tokenized_utts, wordbound, maxlen = self.tokenize(input_text)
        inputs, input_masks = self.get_input_ids(tokenized_utts, maxlen)
        if self.pooling == 'index':
            input_bounds, wordmask = self.get_wordbound_idx(wordbound)
        else:
            input_bounds, wordmask = self.get_wordbound_sum(wordbound, maxlen, input_masks)
        segment_ids = inputs.new_zeros(inputs.size())

        # begin forwarding
        if training:
            outputs = self.model(
                inputs,
                attention_mask=input_masks,
            )

            sequence_output = outputs[0]
            pooled_output = outputs[1]
            wordlevel_out = self.pooling_func(sequence_output, input_bounds)
        else:
            with torch.no_grad():
                outputs = self.model(
                    inputs,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    position_ids=None,
                    head_mask=None
                )
                sequence_output = outputs[0]
                pooled_output = outputs[1]
                wordlevel_out = self.pooling_func(sequence_output, input_bounds)
        return wordlevel_out[:, 1:], wordmask[:, 1:], pooled_output, inputs, sequence_output
