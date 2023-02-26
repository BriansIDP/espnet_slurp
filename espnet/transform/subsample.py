"""Subsampling module for preprocessing."""

import numpy
from espnet.transform.functional import FuncTrans

def subsample(x, sample_rate=3, concat=True, shift=True):
    """subsampling
    
    subsample the input feature
    :param numpy.ndarray x: spectrogram (time, frequency)
    :param int sample_rate: subsampling rate
    :param bool concat: concatenate adjacent frames
    :param bool shift: multiple starting frames
    :returns numpy.ndarray: subsampled features (time/sample_rate, frequency)
        or (time/sample_rate, frequency, sample_rate)
    """
    if shift:
        input_features = [x[i:] for i in range(sample_rate)]
    else:
        input_features = [x]
    if concat:
        feature_dim = input_features[0].shape[1]
        end_frames = [int(numpy.floor(input_feature.shape[0]/sample_rate)*sample_rate)
                      for input_feature in input_features]
        input_features = [input_feature[:end_frames[i]].reshape(-1, sample_rate*feature_dim)
                          for i, input_feature in enumerate(input_features)]
    else:
        input_features = [input_feature[::sample_rate] for input_feature in input_features]
    if shift:
        feature_lens = [input_feature.shape[0] for input_feature in input_features]
        min_len = min(feature_lens)
        input_features = numpy.stack([input_feature[:min_len] for input_feature in input_features],
                                     axis=-1)
    else:
        input_features = input_features[0]
    return input_features


class Subsample(FuncTrans):
    _func = subsample
    __doc__ = subsample.__doc__
    def __call__(self, x, train):
        shift = self.kwargs['shift']
        if not train:
            self.kwargs['shift'] = False
        new_x = super().__call__(x)
        self.kwargs['shift'] = shift
        return new_x
