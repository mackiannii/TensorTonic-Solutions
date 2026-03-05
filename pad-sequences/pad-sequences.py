import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # lenghts of all of the arrays in the sequence
    lengths = [len(s) for s in seqs]
    
    # L is the max length that we grab from the lengths
    L = max_len if max_len is not None else max(lengths, default=0)
    
    # padded creates a full array with the shape of (Len(seq), max L)), and then we pad the rest of the values with 0
    padded = np.full((len(seqs), L), pad_value)

    # for the index and the array inside the sequences
    # We set the padded array
    for i, s in enumerate(seqs):
        padded[i, :min(len(s), L)] = s[:L]

    return padded