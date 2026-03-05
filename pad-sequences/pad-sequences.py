import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    
    # Determine the target sequence length L
    # If max_len is not provided, use the longest sequence length
    if max_len is None:
        L = max(len(s) for s in seqs) if seqs else 0
    else:
        L = max_len

    # Iterate through each sequence and adjust it to length L
    for i, s in enumerate(seqs):

        # Current sequence length
        Li = len(s)

        # Amount of padding needed (if sequence is shorter than L)
        # max(...) prevents negative values
        padding_needed = max(0, L - Li)

        # Amount of truncation needed (if sequence is longer than L)
        truncation_needed = max(0, Li - L)

        # If the sequence is too long, truncate it
        if truncation_needed:
            seqs[i] = s[:L]

        # If the sequence is too short, pad it with pad_value
        elif padding_needed:
            seqs[i] = s + [pad_value] * padding_needed

    return np.array(seqs)