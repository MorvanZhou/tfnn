import numpy as np
import pandas as pd

def next_batch(data, batch_size, loop=False):
    try:
        data._batch_index_segments
    except AttributeError:
        data._n_segments = data.n_samples // batch_size
        data._batch_index_segments = np.array_split(np.arange(data.n_samples), data._n_segments)
        data._batch_counter = 0
    if loop:
        data._batch_counter += 1
        _mini_batch_indexes = data._batch_index_segments[data._batch_counter - 1]
        batch_xs = data.xs.iloc[_mini_batch_indexes, :]
        batch_ys = data.ys.iloc[_mini_batch_indexes, :]
        if data._batch_counter == data._n_segments:
            data._batch_counter = 0
    else:
        data._batch_counter += 1
        _mini_batch_indexes = data._batch_index_segments[data._batch_counter - 1]
        batch_xs = data.xs.iloc[_mini_batch_indexes, :]
        batch_ys = data.ys.iloc[_mini_batch_indexes, :]
        if data._batch_counter == data._n_segments:
            raise IndexError('No more training data. Either set loop=True or reduce training steps.')
    return [batch_xs, batch_ys]