# word2gauss
Gaussian word embeddings

Python/Cython implementation of [http://arxiv.org/abs/1412.6623](Luke
Vilnis and Andrew McCallum
<i>Word Representations via Gaussian Embedding</i>, ICLR 2015)
that represents each word as a multivariate Gaussian.
Scales to large corpora using Cython extensions and threading with asynchronous
stochastic gradient descent (Adagrad).

## Getting started

### Installing
TODO

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
sampling needed to generate training positive/negative pairs from a sentence
(see details below).

### Learning embeddings

```python
from word2gauss import GaussianEmbedding, text_to_pairs
from vocab import Vocabulary

# the Vocabulary class has this interface:
# vocab.tokenize(text) returns a numpy array of token IDs in the text
#   or -1 for OOV tokens
# vocab.random_ids(N) returns a length N numpy array of random IDs
vocab = Vocabulary(...)

# TODO - loop through corpus, batch documents, call train
```


## Background details
Instead of representing a word as a vector as in `word2vec`, `word2gauss`
represents each word as a multivariate Gaussian.  Assume some dictionary
of known tokens `w[i], i = 1 .. N`, we represent each word with
a probability `P[i]`, a `K` dimensional Gaussian parameterized by
```
   P[i] ~ N(x; mu[i], Sigma[i])
```
Here, `mu[i]` and `Sigma[i]` are the mean and co-variance matrix
for word `i`.  The mean is a vector of length `K` and in most general
case `Sigma[i]` is a `(K, K)` matrix.  The paper makes one of two
approximations to simply `Sigma[i]`:

* 'diagonal' in which case `Sigma[i]` is a vector length `K`
* 'spherical' in which case `Sigma[i]` is a single float

To learn the probabilities, first define an energy function 
`E(P[i], P[j])` that returns a similarity like measure of the two
probabilities.  Given a pair of "positive" and "negative" indices,
define `Delta E = E(P[posi], P[posj]) - E(P[negi], P[negj])`.
Intuitively the training process optimizes the parameters
to make `Delta E` positive.  Formally, use a max-margin loss:
```
    loss = max(0, Closs - Delta E)
```
and optimize the parameters to minimize the sum of the loss over
the entire training set of positive/negative pairs.


