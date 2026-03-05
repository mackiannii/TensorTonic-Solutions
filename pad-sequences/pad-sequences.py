import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    L = max_len if max_len is not None else max((len(s) for s in seqs), default=0)    
    arr = np.full((len(seqs), L), pad_value)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s[:L]
    return arr