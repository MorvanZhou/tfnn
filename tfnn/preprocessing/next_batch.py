import numpy as np


def next_batch(data, batch_size):
    batch_data = data.copy()
    if not hasattr(data, '_batch_loop_counter'):
        data._batch_loop_counter = 0
    else:
        data._batch_loop_counter += 1
    indices = np.arange(data._batch_loop_counter*batch_size,
                        (data._batch_loop_counter+1)*batch_size) % len(data.data)
    batch_data.data = data.data[indices, :]
    return batch_data
