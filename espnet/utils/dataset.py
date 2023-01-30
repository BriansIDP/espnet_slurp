#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""pytorch dataset and dataloader implementation for chainer training."""

import torch
import torch.utils.data


class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1]

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])


class TransformDataset(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """

    def __init__(self, data, transform):
        """Init function."""
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        return self.transform(self.data[idx])


class ChainerDataLoader(object):
    """Pytorch dataloader in chainer style.

    Args:
        all args for torch.utils.data.dataloader.Dataloader

    """

    def __init__(self, **kwargs):
        """Init function."""
        if 'sampler' in kwargs and kwargs['sampler']:
            self.sampler = ResumableRandomSampler(kwargs['dataset'])
            kwargs['sampler'] = self.sampler
            kwargs['shuffle'] = False
        elif 'sampler' in kwargs and not kwargs['sampler']:
            kwargs.pop('sampler')
        self.loader = torch.utils.data.dataloader.DataLoader(**kwargs)
        self.len = len(kwargs["dataset"])
        self.current_position = 0
        self.epoch = 0
        self.iter = None
        self.kwargs = kwargs

    def next(self):
        """Implement next function."""
        if self.iter is None:
            self.iter = iter(self.loader)
        try:
            ret = next(self.iter)
        except StopIteration:
            self.iter = None
            return self.next()
        self.current_position += 1
        if self.current_position == self.len:
            self.epoch = self.epoch + 1
            self.current_position = 0
        return ret

    def __iter__(self):
        """Implement iter function."""
        for batch in self.loader:
            yield batch

    @property
    def epoch_detail(self):
        """Epoch_detail required by chainer."""
        return self.epoch + self.current_position / self.len

    def serialize(self, serializer):
        """Serialize and deserialize function."""
        epoch = serializer("epoch", self.epoch)
        current_position = serializer("current_position", self.current_position)
        self.epoch = epoch
        self.current_position = current_position

    def start_shuffle(self):
        """Shuffle function for sortagrad."""
        self.kwargs["shuffle"] = True
        if "sampler" in self.kwargs and not self.kwargs['sampler']:
            self.kwargs.pop('sampler')
        elif "sampler" in self.kwargs and self.kwargs['sampler']:
            self.kwargs["shuffle"] = False
        self.loader = torch.utils.data.dataloader.DataLoader(**self.kwargs)

    def finalize(self):
        """Implement finalize function."""
        del self.loader
