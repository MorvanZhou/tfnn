import numpy as np


def onehot_encode(seqs, k_1=False):
    seqs_copy = seqs.copy()
    seqs_copy = _onehot_seq(seqs_copy, k_1)
    return seqs_copy


def _onehot_seq(seq, k_1=False):
    unique_num, indices = np.unique(seq, return_inverse=True)
    n_samples = len(seq)
    n_classes = len(unique_num)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), indices] = 1
    return one_hot
