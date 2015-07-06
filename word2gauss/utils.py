import numpy as np

def cosine(a, b, normalize=True):
    '''
    Compute the cosine measure between a and b.

    a = numpy array with each row one observation.
    b = vector or 1d array, len(b) == a.shape[1]

    Returns length a.shape[0] array with the cosine similarities
    '''
    if not normalize:
        return np.dot(a, b.reshape(-1, 1)).flatten()

    else:
        # normalize b
        norm_b = np.sqrt(np.sum(b ** 2))
        b_normalized = b / norm_b

        # get norms of a
        norm_a = np.sqrt(np.sum(a ** 2, axis=1))

        # compute cosine measure and normalize
        return np.dot(a, b_normalized.reshape(-1, 1)).flatten() / norm_a

