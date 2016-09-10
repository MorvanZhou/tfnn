import numpy as np


def next_batch(data, batch_size):
    if not hasattr(data, '_batch_loop_counter'):
        data._batch_loop_counter = 0
    else:
        data._batch_loop_counter += 1
    indices = np.arange(data._batch_loop_counter*batch_size,
                        (data._batch_loop_counter+1)*batch_size) % data.n_samples
    return [data.xs[indices, :], data.ys[indices, :]]
