# word2gauss
Gaussian word embeddings

Python/Cython implementation of [Luke Vilnis and Andrew McCallum
<i>Word Representations via Gaussian Embedding</i>, ICLR 2015](http://arxiv.org/abs/1412.6623)
that represents each word as a multivariate Gaussian.
Scales to (relatively) large corpora using Cython extensions and threading
with asynchronous stochastic gradient descent (Adagrad).

## Getting started

### Installing
First install numpy, scipy and the packages in `requirements.txt`.
Then `sudo make install`.  `make test` runs the test suite.

NOTE: The performance sensitive parts of the code have been carefully
written in a way that allows gcc to auto-vectorize all the important loops.
Accordingly we recommend using gcc to compile and setting these
flags for building:
```
export CFLAGS="-ftree-vectorizer-verbose=2 -O3 -ffast-math"
sudo -E bash -c "make install"
```

If you are using a Mac, gcc compiled code runs approximately 2.5X faster than
the default clang compiler.  You can force the build to use gcc instead
of clang with:
```
# change these to the location of gcc -- note that /usr/bin/gcc is really
# clang in a default XCode installation
export CC=/usr/local/bin/gcc
export CXX=/usr/local/bin/g++
export CFLAGS="-ftree-vectorizer-verbose=2 -O3 -ffast-math"
sudo -E bash -c "make install"
```


### Code overview
The `GaussianEmbedding` class is the main workhorse for most tasks.  It
stores the model data, deals with serialization to/from files and
learns the parameters.  To allow embedding of non-word types like
hierarchies and entailment relations, `GaussianEmbedding` has
no knowledge of any vocabulary and operates only on
unique IDs.  Each ID is a `uint32` from `0 .. N-1` with `-1` signifying
an OOV token.

For learning word embeddings, the token - id mapping is off-loaded to a
`Vocabulary` class that translates streams of documents into `token_id` lists
and provides the ability to draw a random `token_id` from the
token distribution.  The random generator is necessary for the negative
sampling needed to generate training positive/negative training pairs from a
sentence (see details below).  We assume the `Vocabulary` has these methods:

* `tokenize(text)` returns a numpy array of token IDs in the text or -1 for OOV tokens
* `random_ids(N)` returns a length N numpy array of random IDs
* `word2id(string)` and `id2word(uint32)` map from string <-> id
* `__len__(self)` returns number of words in vocabulary


### Learning embeddings

To learn embeddings, you will need a suitable corpus and an implementation
of the `Vocabulary` interface.

```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gzip import GzipFile

from word2gauss import GaussianEmbedding, iter_pairs
from vocab import Vocabulary

# load the vocabulary
vocab = Vocabulary(...)

# create the embedding to train
# use 100 dimensional spherical Gaussian with KL-divergence as energy function
embed = GaussianEmbedding(len(vocab), 100,
    covariance_type='spherical', energy_type='KL')

# open the corpus and train with 8 threads
# the corpus is just an iterator of documents, here a new line separated
# gzip file for example
with GzipFile('location_of_corpus', 'r') as corpus:
    embed.train(iter_pairs(corpus, vocab), n_workers=8)

# save the model for later
embed.save('model_file_location', vocab=vocab.id2word, full=True)
```

### Examining trained models
```python
from word2gauss import GaussianEmbedding
from vocab import Vocabulary

# load in a previously trained model and the vocab
vocab = Vocabulary(...)
embed = GaussianEmbedding.load('model_file_location')

# find nearest neighbors to 'rock'
embed.nearest_neighbors('rock', vocab=vocab)

# find nearest neighbors to 'rock' sorted by covariance
embed.nearest_neighbors('rock', num=100, vocab=vocab, sort_order='sigma')

# solve king + woman - man = ??
embed.nearest_neighbors([['king', 'woman'], ['man']], num=10, vocab=vocab)
```


## Background details
Instead of representing a word as a vector as in `word2vec`, `word2gauss`
represents each word as a multivariate Gaussian.  Assuming some dictionary
of known tokens `w[i], i = 0 .. N-1`, each word is represented as
a probability `P[i]`, a `K` dimensional Gaussian parameterized by
```
   P[i] ~ N(x; mu[i], Sigma[i])
```
Here, `mu[i]` and `Sigma[i]` are the mean and co-variance matrix
for word `i`.  The mean is a vector of length `K` and in the most general
case `Sigma[i]` is a `(K, K)` matrix.  The paper makes one of two
approximations to simplify `Sigma[i]`:

* 'diagonal' in which case `Sigma[i]` is a vector length `K`
* 'spherical' in which case `Sigma[i]` is a single float

To learn the probabilities, first define an energy function
`E(P[i], P[j])` that returns a similarity like measure of the two
probabilities.  Both the symmetric Expected Likelihood Inner Product
and asymmetric KL-divergence are implemented.

Given a pair of "positive" and "negative" indices,
define `Delta E = E(P[posi], P[posj]) - E(P[negi], P[negj])`.
Intuitively the training process optimizes the parameters
to make `Delta E` positive.  Formally, use a max-margin loss:
```
    loss = max(0, Closs - Delta E)
```
and optimize the parameters to minimize the sum of the loss over
the entire training set of positive/negative pairs.

To generate the training pairs, use co-occuring words as the positive
examples and a randomly sampled words as the negative examples.
Since the energy function is potentially asymmetric, for each co-occuring
word pair randomly sample both the left and right tokens for negative
examples.  In addition, we allow the option to generate several
sets of training pairs from each word.
In pseudo-code:
```
for sentence in corpus:
    for i in len(sentence):
        for k in 1..window_size:
            for nsample in 1..number_of_samples_per_word:
                positive pair = (left, right) = (sentence[i], sentence[i + k])
                negative pairs = [(left, random ID), (random ID, right)]
                update model weights
```

