
from itertools import islice

from .embeddings import text_to_pairs


def iter_pairs(fin, vocab, batch_size=10, nsamples=2, window=5):
    '''
    Convert a document stream to batches of pairs used for training embeddings.

    iter_pairs is a generator that yields batches of pairs that can
    be passed to GaussianEmbedding.train

    fin = an iterator of documents / sentences (e.g. a file like object)
        Each element is a string of raw text
    vocab = something implementing the Vocabulary interface
    batch_size = size of batches

    window = Number of words to the left and right of center word to include
        as positive pairs
    nsamples = number of negative samples to drawn for each center word
    '''
    documents = iter(fin)
    batch = list(islice(documents, batch_size))
    while len(batch) > 0:
        text = [
            vocab.tokenize_ids(doc, remove_oov=False)
            for doc in batch
        ]
        pairs = text_to_pairs(text, vocab.random_ids,
            nsamples_per_word=nsamples,
            half_window_size=window)
        yield pairs
        batch = list(islice(documents, batch_size))

