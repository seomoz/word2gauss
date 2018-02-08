#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True

# Two different energies:
#
# Have inner product based energy (Expected Likelihood, EL)
#
#   E(P[i], P[j]) = N(0; mu[i] - mu[j], Sigma[i] + Sigma[j])
#   dE/dmu[i] = - dE/dmu[j] = -Delta[i, j]
#   dE/dSigma[i] = dE/dSigma[j] = 0.5 * (
#       Delta[i, j] * Delta[i, j].T - (Sigma[i] + Sigma[j]) ** -1)
#   Delta[i, j] = ((Sigma[i] + Sigma[j]) ** -1) * (mu[i] - mu[j])
#
# Also have KL-div based energy:
#   E(P[i], P[j]) = 0.5 * (
#       Tr((Sigma[i] ** -1) * Sigma[j]) +
#       (mu[i] - mu[j]).T * (Sigma[i] ** -1) * (mu[i] - mu[j]) -
#       d - log(det(Sigma[j]) / det(Sigma[i]))
#   )
#   dE/dmu[i] = - dE/dmu[j] = -Delta'[i, j]
#   dE/Sigma[i] = 0.5 * (
#       (Sigma[i] ** -1) * Sigma[j] * (Sigma[i] ** -1) +
#       Delta'[i, j] * Delta'[i, j].T -
#       (Sigma[i] ** -1)
#   )
#   dE/dSigma[j] = 0.5 * ((Sigma[j] ** -1) - (Sigma[i] ** -1))
#  Delta'[i, j] = (Sigma[i] ** -1) * (mu[i] - mu[j])
#
# Have Loss:
#   Loss = SUM_[all positive/negative pairs] OF
#       max(0, C - E(P[i_pos], P[j_pos]) + E(P[i_neg, P[i_neg]))
#   dLoss/dparameters = dLoss / dE * dE/dparameters
#   dLoss/dparameters = 0 if loss == 0
#   dLoss/dparamters = -1 * dE(P[i_pos], P[j_pos]) / dparamters +
#       dE(P[i_neg], P[j_neg])
#

# NOTE: C/C++ is row major (so A[i, :] is contiguous)

cimport cython
cimport numpy as np

np.import_array()

from libc.math cimport log, sqrt
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string

import logging
import time
import json
import numpy as np

from tarfile import open as open_tar
from contextlib import closing

from .utils import cosine

from cpython.version cimport PY_MAJOR_VERSION
import six

from six.moves.queue import Queue

LOGGER = logging.getLogger()


cdef extern from "stdint.h":
    ctypedef unsigned long long uint32_t
cdef uint32_t UINT32_MAX = 4294967295

# type of our floating point arrays
cdef type DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# types of covariance matrices we support
cdef uint32_t SPHERICAL = 1
cdef uint32_t DIAGONAL = 2
COV_MAP = {1: 'spherical', 2: 'diagonal'}

# define a generic interface for energy and gradient functions
# (e.g. symmetric or KL-divergence).  This defines a type for the function...

# in the energy / gradient functions, here is the encoding for
# center_index:
#  0 : i is center word, j is context
#  1 : j is center word, i is context
#  2 : both i and j are "center" (use center embeddings -- called
#      only in wrappers)

# interface for an energy function
# energy(i, j, # word indices to compute energy for
#        center_index,
#        mu_ptr,  # pointer to mu array
#        sigma_ptr,  # pointer to sigma array
#        covariance_type (SPHERICAL, etc)
#        nwords, ndimensions)
ctypedef DTYPE_t (*energy_t)(size_t, size_t, size_t,
                             DTYPE_t*, DTYPE_t*, uint32_t, size_t, size_t) nogil

# interface for a gradient function
# gradient(i, j, # word indices to compute energy for
#          center_index,
#          dEdmui_ptr, dEdsigma1_ptr, dEdmuj_ptr, dEdsigmaj_ptr
#          mu_ptr,  # pointer to mu array
#          sigma_ptr,  # pointer to sigma array
#          covariance_type (SPHERICAL, etc)
#          nwords, ndimensions)
ctypedef void (*gradient_t)(size_t, size_t, size_t,
                            DTYPE_t*, DTYPE_t*, DTYPE_t*, DTYPE_t*,
                            DTYPE_t*, DTYPE_t*, uint32_t, size_t, size_t) nogil

# learning rates.  We'll store global learning rates for mu and sigma,
# as well as minimum rates for each
cdef struct LearningRates:
    DTYPE_t mu, sigma, mu_min, sigma_min

# fix to accept Python3 unicode strings
ctypedef string char_type
cdef char_type _chars(s):
    if isinstance(s, unicode):
        s = (<unicode> s).encode('ascii')
    return s

cdef class GaussianEmbedding:
    '''
    Represent words as gaussian distributions

    Holds model parameters, deals with serialization and learning.

    We learn separate embeddings for the focus and context words,
    similar to word2vec.  Each set of parameters (mu, sigma, etc)
    are all stored as one matrix where the first N rows correspond
    to the focus parameters and the second N rows correspond
    to the context parameters.

    Attributes:
        N = number of words
        K = dimension of each representation
        mu = (2N, K)
        sigma = either (2N, 1) array (for spherical) or (2N, K) (for diagonal)
        acc_grad_mu = (2N, K) array, the accumulated gradients for each
            mu update
        acc_grad_sigma = (2N, 1) or (2N, K) array with the accumulated
            gradients for each sigma

    Since we use Cython we also store the C pointer to the underlying
    raw data.
    '''

    # the numpy arrays holding data as described above
    # NOTE: the Cython code assumes the data is stored in C contiguous order
    # and asserted on creation
    cdef np.ndarray mu, sigma, acc_grad_mu, acc_grad_sigma

    # the covariance type as defined above
    cdef uint32_t covariance_type

    # either 'IP' or 'KL'

    cdef char_type energy_type

    # number of words
    cdef size_t N
    # dimension of vectors
    cdef size_t K

    # L2 regularization parameters
    # mu_max = max L2 norm of mu
    cdef DTYPE_t mu_max
    # min and max L2 norm of Sigma (= max/min entries on the diagonal)
    cdef DTYPE_t sigma_min, sigma_max

    # learning rates
    cdef LearningRates eta

    # the Closs in max-margin function
    cdef DTYPE_t Closs

    # energy and gradient functions
    cdef energy_t energy_func
    cdef gradient_t gradient_func

    # pointers to data arrays
    cdef DTYPE_t *mu_ptr
    cdef DTYPE_t *sigma_ptr
    cdef DTYPE_t *acc_grad_mu_ptr
    cdef DTYPE_t *acc_grad_sigma_ptr

    def __cinit__(self, N, size=100,
                  covariance_type='spherical', mu_max=2.0, sigma_min=0.7, sigma_max=1.5,
                  energy_type='KL',
                  init_params={
                      'mu0': 0.1,
                      'sigma_mean0': 0.5,
                      'sigma_std0': 1.0
                  },
                  eta=0.1, Closs=0.1,
                  mu=None, sigma=None):
        '''
        N = number of distributions (e.g. number of words)
        size = dimension of each Gaussian
        covariance_type = 'spherical' or 'diagonal'
        energy_type = 'KL' or 'IP'
        mu_max = maximum L2 norm of each mu
        sigma_min, sigma_max = maximum/min eigenvalues of sigma
        init_params = {
            mu0: initial means are random normal w/ mean=0, std=mu0
            sigma_mean0, sigma_std0: initial sigma is random normal with
                mean=sigma_mean0, std=sigma_std0
            NOTE: these are ignored if mu/sigma is explicitly specified
        }
        eta = learning rates.  Two options:
            * pass a single float which gives the global learning rate
            for both mu and sigma with no minimum
            * pass dict with keys mu and sigma (global learning rate
                for mu / sigma) and mu_min, sigma_min (minimum local
                learning rate for each)
        Closs = regularization parameter in max-margin loss
            loss = max(0.0, Closs - energy(pos) + energy(neg)
        if mu/sigma are not None then they specify the initial values
            for the parameters
        '''
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _sigma
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _acc_grad_mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _acc_grad_sigma

        if covariance_type == 'spherical':
            self.covariance_type = SPHERICAL
        elif covariance_type == 'diagonal':
            self.covariance_type = DIAGONAL
        else:
            raise ValueError

        self.energy_type = _chars(energy_type)
        # self.energy_type = energy_type
        if energy_type == 'KL':
            self.energy_func = <energy_t> kl_energy
            self.gradient_func = <gradient_t> kl_gradient
        elif energy_type == 'IP':
            self.energy_func = <energy_t> ip_energy
            self.gradient_func = <gradient_t> ip_gradient
        else:
            raise ValueError

        self.N = N
        self.K = size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu_max = mu_max
        self.Closs = Closs

        if isinstance(eta, dict):
            # NOTE: cython automatically converts from struct to dict
            self.eta = eta
        else:
            self.eta.mu = eta
            self.eta.sigma = eta
            self.eta.mu_min = 0.0
            self.eta.sigma_min = 0.0

        # Initialize parameters
        np.random.seed(5)

        if mu is None:
            _mu = init_params['mu0'] * np.ascontiguousarray(
                np.random.randn(2 * self.N, self.K).astype(DTYPE))
        else:
            assert mu.shape[0] == 2 * N
            assert mu.shape[1] == self.K
            _mu = mu
        self.mu = _mu

        if sigma is None:
            if self.covariance_type == SPHERICAL:
                s = np.random.randn(2 * self.N, 1).astype(DTYPE)
            elif self.covariance_type == DIAGONAL:
                s = np.random.randn(2 * self.N, self.K).astype(DTYPE)
            s *= init_params['sigma_std0']
            s += init_params['sigma_mean0']
            _sigma = np.ascontiguousarray(
                np.maximum(sigma_min, np.minimum(s, sigma_max)))
        else:
            assert sigma.shape[0] == 2 * N
            if self.covariance_type == SPHERICAL:
                assert sigma.shape[1] == 1
            elif self.covariance_type == DIAGONAL:
                assert sigma.shape[1] == self.K
            _sigma = sigma
        self.sigma = _sigma

        assert _mu.flags['C_CONTIGUOUS']
        assert _sigma.flags['C_CONTIGUOUS']
        self.mu_ptr = &_mu[0, 0]
        self.sigma_ptr = &_sigma[0, 0]

        # working space for accumulated gradients
        _acc_grad_mu = np.ascontiguousarray(
            np.zeros((2 * N, self.K), dtype=DTYPE))
        assert _acc_grad_mu.flags['C_CONTIGUOUS']
        self.acc_grad_mu = _acc_grad_mu
        self.acc_grad_mu_ptr = &_acc_grad_mu[0, 0]

        if self.covariance_type == SPHERICAL:
            _acc_grad_sigma = np.ascontiguousarray(
                np.zeros((2 * N, 1), dtype=DTYPE))
        elif self.covariance_type == DIAGONAL:
            _acc_grad_sigma = np.ascontiguousarray(
                np.zeros((2 * N, self.K), dtype=DTYPE))
        assert _acc_grad_sigma.flags['C_CONTIGUOUS']
        self.acc_grad_sigma = _acc_grad_sigma
        self.acc_grad_sigma_ptr = &_acc_grad_sigma[0, 0]

    def update(self, N, init_params={
        'mu0': 0.1,
        'sigma_mean0': 0.5,
        'sigma_std0': 1.0
    }):
        '''
        This function adds to the current np arrays: mu, sigma, acc_grad_mu
        and acc_grad_sigma to account for any new vocabulary that we need to
         introduce to the model for retraining.

        If the original size of the vocabulary is N, and the new size is M
        It adds new rows to self.mu to make it 2 * M x K
        It increases the size of the other 2 * M x 1 dimensional arrays.

        N = Total size of the vocabulary
        init_params = initialization parameters to initialize new rows
        '''
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _sigma
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _acc_grad_mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _acc_grad_sigma

        LOGGER.info("New vocab size %i" % N)
        LOGGER.info("Old vocab size %i" % self.N)
        if N > self.N:
            # there are new words in the vocabulary
            # add more rows
            np.random.seed(12345)

            n_words = N - self.N
            LOGGER.info("%i new words" % n_words)

            # update mu with n_words
            _mu = np.ascontiguousarray(np.vstack([
                self.mu[:self.N, :],
                init_params['mu0'] * np.random.randn(
                    n_words, self.K).astype(DTYPE),
                self.mu[self.N:, :],
                init_params['mu0'] * np.random.randn(
                    n_words, self.K).astype(DTYPE),
            ]))
            self.mu = _mu
            assert _mu.flags['C_CONTIGUOUS'] and _mu.flags['OWNDATA']
            self.mu_ptr = &_mu[0, 0]

            # update sigma
            if self.covariance_type == SPHERICAL:
                s = np.random.randn(2 * n_words, 1).astype(DTYPE)
            elif self.covariance_type == DIAGONAL:
                s = np.random.randn(2 * n_words, self.K).astype(DTYPE)
            s *= init_params['sigma_std0']
            s += init_params['sigma_mean0']
            s = np.maximum(self.sigma_min, np.minimum(s, self.sigma_max))
            _sigma = np.ascontiguousarray(np.vstack([
                self.sigma[:self.N, :],
                s[:n_words, :],
                self.sigma[self.N:, :],
                s[n_words:, :]
            ]))
            self.sigma = _sigma
            assert _sigma.flags['C_CONTIGUOUS'] and _sigma.flags['OWNDATA']
            self.sigma_ptr = &_sigma[0, 0]

            # update acc_grad_mu
            _acc_grad_mu = np.ascontiguousarray(np.vstack([
                self.acc_grad_mu[:self.N, :],
                np.zeros((n_words, self.K), dtype=DTYPE),
                self.acc_grad_mu[self.N:, :],
                np.zeros((n_words, self.K), dtype=DTYPE)
            ]))
            self.acc_grad_mu = _acc_grad_mu
            assert _acc_grad_mu.flags['C_CONTIGUOUS'] and \
                   _acc_grad_mu.flags['OWNDATA']
            self.acc_grad_mu_ptr = &_acc_grad_mu[0, 0]

            # update acc_grad_sigma
            if self.covariance_type == SPHERICAL:
                sigma_dim = 1
            elif self.covariance_type == DIAGONAL:
                sigma_dim = self.K
            _acc_grad_sigma = np.ascontiguousarray(np.vstack([
                self.acc_grad_sigma[:self.N, :],
                np.zeros((n_words, sigma_dim), dtype=DTYPE),
                self.acc_grad_sigma[self.N:, :],
                np.zeros((n_words, sigma_dim), dtype=DTYPE)
            ]))
            self.acc_grad_sigma = _acc_grad_sigma
            assert _acc_grad_sigma.flags['C_CONTIGUOUS'] and \
                   _acc_grad_sigma.flags['OWNDATA']
            self.acc_grad_sigma_ptr = &_acc_grad_sigma[0, 0]

            self.N = N

    def save(self, fname, vocab=None, full=True):
        '''
        Writes a gzipped text file of the model in word2vec text format

        word_or_id1 0.5 -0.2 1.0 ...
        word_or_id2 ...

        This class doesn't have knowledge of id -> word mapping, so
        it can be passed in as a callable vocab(id) return the word string

        if full=True, then writes out the full model.  It is a tar.gz
        file with the word vectors and Sigma files, in addition
        to files for the other state (accumulated gradient for
            training, model parameters)

        It has files:
            word_mu: the word2vec file with word/id and mu vectors of
                center words.
            mu_context: mu vectors for context embeddings, one per line
            sigma: the sigma for each vector, one per line.  The first
                N lines correspond to sigma for the center words and
                the next N lines correspond to sigma for the context words
            acc_grad_mu and acc_grad_sigma: one value per line/word with
                accumulated gradient sums
            parameters: json file with model parameters (hyperparameters, etc)
        The wordid is implicitly defined by the word_mu file, then
            assumed to align across the other files
        '''
        from gzip import GzipFile
        from tempfile import NamedTemporaryFile

        if not vocab:
            vocab = lambda x: x

        def write_word2vec(fout):
            for i in range(self.N):
                line = [vocab(i)] + self.mu[i, :].tolist()
                to_write = ' '.join('%s' % ele for ele in line) + '\n'
                if PY_MAJOR_VERSION >= 3:
                    to_write = to_write.encode('utf-8')

                fout.write(to_write)

        def save_array(a, name, fout):
            with NamedTemporaryFile() as tmp:
                np.savetxt(tmp, a, fmt='%s')
                tmp.seek(0)
                fout.add(tmp.name, arcname=name)

        if not full:
            with GzipFile(fname, 'w') as fout:
                write_word2vec(fout)
            return

        # otherwise write the full file
        # easiest way to manage is to write out to temp files then
        # add to archive
        with open_tar(fname, 'w:gz') as fout:
            # word2vec file
            with NamedTemporaryFile() as tmp:
                write_word2vec(tmp)
                tmp.seek(0)
                fout.add(tmp.name, arcname='word_mu')

            # context mu
            save_array(self.mu[self.N:, :], 'mu_context', fout)

            # sigma, accumulated gradient
            save_array(self.sigma, 'sigma', fout)
            save_array(self.acc_grad_mu, 'acc_grad_mu', fout)
            save_array(self.acc_grad_sigma, 'acc_grad_sigma', fout)

            # parameters
            params = {
                'N': self.N,
                'K': self.K,
                'covariance_type': self.covariance_type,
                'energy_type': self.energy_type,
                'sigma_min': self.sigma_min,
                'sigma_max': self.sigma_max,
                'mu_max': self.mu_max,
                'eta': self.eta,
                'Closs': self.Closs
            }

            if PY_MAJOR_VERSION >= 3:
                params['energy_type'] = _ustring(params['energy_type'])

            with NamedTemporaryFile() as tmp:
                json_dump = json.dumps(params)
                if PY_MAJOR_VERSION >= 3:
                    json_dump = json_dump.encode('utf-8')

                tmp.write(json_dump)
                tmp.seek(0)
                fout.add(tmp.name, arcname='parameters')

    @classmethod
    def load(cls, fname):
        '''
        Load the model from the saved tar ball (from a previous
            call to save with full=True)
        '''
        # read in the parameters, construct the class, then read
        # data and store it in class
        with open_tar(fname, 'r') as fin:
            with closing(fin.extractfile('parameters')) as f:
                file_contents = f.read()

                if PY_MAJOR_VERSION >= 3:
                    file_contents = _ustring(file_contents)

                params = json.loads(file_contents)

            ret = cls(params['N'], size=params['K'],
                      covariance_type=COV_MAP[params['covariance_type']],
                      mu_max=params['mu_max'],
                      sigma_min=params['sigma_min'], sigma_max=params['sigma_max'],
                      energy_type=params['energy_type'],
                      eta=params['eta'], Closs=params['Closs'])

        ret._data_from_file(fname)

        return ret

    def _data_from_file(self, fname):
        '''
        Set mu, sigma, acc_grad* from the saved file
        '''
        N = self.N
        K = self.K
        N2 = 2 * N

        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _sigma
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _acc_grad_mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _acc_grad_sigma

        # set the data
        with open_tar(fname, 'r') as fin:
            _mu = np.empty((2 * N, K), dtype=DTYPE)
            with closing(fin.extractfile('word_mu')) as f:
                for i, line in enumerate(f):
                    ls = line.strip().split()
                    # ls[0] is the word/id, skip it.  rest are mu
                    mus = ls[1:]
                    if len(mus) < K:
                        # some lines have fewer features than expected
                        # this usually happens when we encode the null string ''
                        # into the model
                        logging.error('error with token {}'.format(line))
                        logging.error('expected line to have {} features, found {}; skipping'.format(
                                        K, len(mus)))
                        continue
                    _mu[i, :] = [float(ele) for ele in mus]
            try:
                #try loading using pandas.read_csv because it's much faster
                from pandas import read_csv

                logging.warn('loading model with pandas.read_csv instead of numpy.loadtxt')
                logging.warn('this is much faster but will result in slightly different values (within a tolerance)')

                with closing(fin.extractfile('mu_context')) as f:
                    _mu[self.N:, :] = read_csv(f, sep="\s+", header=None, \
                        dtype=DTYPE).as_matrix().reshape(N, -1).copy()

                with closing(fin.extractfile('sigma')) as f:
                    _sigma = read_csv(f, sep="\s+", header=None, dtype=DTYPE). \
                    as_matrix().reshape(N2, -1).copy()
                with closing(fin.extractfile('acc_grad_mu')) as f:
                    _acc_grad_mu = read_csv(f, sep="\s+", header=None, \
                        dtype=DTYPE).as_matrix().reshape(N2, K).copy()
                with closing(fin.extractfile('acc_grad_sigma')) as f:
                    _acc_grad_sigma = read_csv(f, sep="\s+", header=None, \
                        dtype=DTYPE).as_matrix().reshape(N2, -1).copy()

            except ImportError:
                #fall back to numpy.loadtext
                with closing(fin.extractfile('mu_context')) as f:
                    _mu[self.N:, :] = np.loadtxt(f, dtype=DTYPE). \
                        reshape(N, -1).copy()

                with closing(fin.extractfile('sigma')) as f:
                    _sigma = np.loadtxt(f, dtype=DTYPE).reshape(N2, -1).copy()
                with closing(fin.extractfile('acc_grad_mu')) as f:
                    _acc_grad_mu = np.loadtxt(f, dtype=DTYPE).reshape(N2, K).copy()
                with closing(fin.extractfile('acc_grad_sigma')) as f:
                    _acc_grad_sigma = np.loadtxt(f, dtype=DTYPE). \
                        reshape(N2, -1).copy()

        self.mu = _mu
        assert self.mu.flags['C_CONTIGUOUS'] and self.mu.flags['OWNDATA']
        self.mu_ptr = &_mu[0, 0]

        self.sigma = _sigma
        assert self.sigma.flags['C_CONTIGUOUS'] and self.sigma.flags['OWNDATA']
        self.sigma_ptr = &_sigma[0, 0]

        self.acc_grad_mu = _acc_grad_mu
        assert (
            self.acc_grad_mu.flags['C_CONTIGUOUS'] and
            self.acc_grad_mu.flags['OWNDATA']
        )
        self.acc_grad_mu_ptr = &_acc_grad_mu[0, 0]

        self.acc_grad_sigma = _acc_grad_sigma
        assert (
            self.acc_grad_sigma.flags['C_CONTIGUOUS'] and
            self.acc_grad_sigma.flags['OWNDATA']
        )
        self.acc_grad_sigma_ptr = &_acc_grad_sigma[0, 0]

    def get_phrases_vector(self, phrases, vocab):
        ''' Input is a list of phrases and the output is a vector
        representation of those phrases
        '''
        vec = np.zeros(self.K)
        # check that the phrases container is not empty
        if not phrases or set(phrases) == {''}:
            return vec
        else:
            for ph in phrases:
                ph_tok = vocab.tokenize(ph)
                phrase_vec = np.zeros(self.K)
                for p in ph_tok:
                    # add a case for when we already have IDs?
                    if isinstance(p, six.string_types):
                        phrase_vec += self.mu[vocab.word2id(p), :]
                if len(ph_tok) != 0:
                    phrase_vec /= len(ph_tok)
                vec += phrase_vec
            vec /= len(phrases)
            return vec

    def phrases_to_vector(self, target, vocab):
        ''' Input is a list of lists, where target[0] is the list of positive
        phrases and target[1] is the optional list of negative phrases
        '''
        positive_vec = self.get_phrases_vector(target[0], vocab=vocab)
        negative_vec = self.get_phrases_vector(target[1], vocab=vocab)
        return (positive_vec - negative_vec)

    def nearest_neighbors(self, target, metric=cosine, num=10, vocab=None,
                          sort_order='similarity'):
        '''Find nearest neighbors.

        target defines the embedding to which we'll find neighbors.  It
        can take a number of different forms depending on use case:

            * if target = uint32 it is interpreted as as word_id
            * if target = [(tuple of uint32), (tuple of uint32)] then
                it is interpreted as a list of "positive" and "negative"
                word_ids.  The first tuple lists positive word_ids and
                the second negative word_ids
            * if target = string then vocab is used to find the corresponding
                word_id
            * if target = [(tuple of strings), (tuple of strings)] then
                it is interpreted as positive and negative words
            * if target = [numpy array], then it is interpreted as a vector
            ** NOTE: if using tuples and only have one entry in the list,
                then need to use the trailing comma syntax, e.g.
                [('king', 'woman'), ('man', )] not [('king', 'woman'), ('man')]

        Metric is callable with this interface:
            array(N) with similarities = metric(
                array(N, K) of all vectors, array(K) of word_id
            )
        Default is cosine similarity.

        vocab: if provided, is an object with callable methods: word2id and
            id2word translate from strings to word_ids and vice versa.

        sort_order = either 'similarity' or 'sigma'.  If 'similarity'
            then sorts results by descending metric, if 'sigma' then
            sort results by increasing sigma
        '''
        # find the distribution to compare similarity to
        if isinstance(target, six.string_types):
            t = self.mu[vocab.word2id(target), :]
        elif isinstance(target, list):
            # positive and negative indices
            t = np.zeros(self.K)
            for pos_neg, fac in zip(target, [1.0, -1.0]):
                for word_or_id in pos_neg:
                    if isinstance(word_or_id, six.string_types):
                        # input is word, get its ID
                        word_id = vocab.word2id(word_or_id)
                        mu_val = self.mu[word_id, :]
                    elif isinstance(word_or_id, int):
                        # input is ID
                        word_id = word_or_id
                        mu_val = self.mu[word_id, :]

                    t += fac * mu_val
        elif "numpy" in str(type(target)):
            t = target
        else:
            t = self.mu[target, :]

        # get the neighbors to the focus words (top half of self.mu)
        scores = metric(self.mu[:self.N, :], t.reshape(1, -1))
        top_indices = scores.argsort()[::-1][:num]
        ret = [
            {
                'id': ind,
                'similarity': scores[ind],
                'sigma': self.sigma[ind, :].prod()
            }
            for ind in top_indices
            ]
        if vocab is not None:
            for k, ind in enumerate(top_indices):
                ret[k]['word'] = vocab.id2word(ind)
        if sort_order == 'sigma':
            ret.sort(key=lambda x: x['sigma'])

        return ret

    def __getattr__(self, name):
        if name == 'mu':
            return self.mu
        elif name == 'sigma':
            return self.sigma
        elif name == 'acc_grad_mu':
            return self.acc_grad_mu
        elif name == 'acc_grad_sigma':
            return self.acc_grad_sigma
        elif name == 'covariance_type':
            return self.covariance_type
        elif name == 'energy_type':
            return self.energy_type
        elif name == 'N':
            return self.N
        elif name == 'K':
            return self.K
        elif name == 'eta':
            return self.eta
        else:
            raise AttributeError

    def train(self, iter_pairs, n_workers=1, reporter=None, report_interval=10):
        '''
        Train the model from an iterator of many batches of pairs.

        use n_workers many workers
        report_interval: report progress every this many batches,
            if None then never report
        if reporter is not None then it is called reporter(self, batch_number)
            every time the report is run
        '''
        # threadpool implementation of training

        from threading import Thread, Lock

        # each job is a batch of pairs from the iterator
        # add jobs to a queue, workers pop from the queue
        # None means no more jobs
        jobs = Queue(maxsize=2 * n_workers)

        # number processed, next time to log, logging interval
        # make it a list so we can modify it in the thread w/o a local var
        processed = [0, report_interval, report_interval]
        t1 = time.time()
        lock = Lock()
        def _worker():
            while True:
                pairs = jobs.get()
                if pairs is None:
                    # no more data
                    break
                self.train_batch(pairs)
                with lock:
                    processed[0] += 1
                    if processed[1] and processed[0] >= processed[1]:
                        t2 = time.time()
                        LOGGER.info("Processed %s batches, elapsed time: %s"
                                    % (processed[0], t2 - t1))
                        processed[1] = processed[0] + processed[2]
                        if reporter:
                            reporter(self, processed[0])

        # start threads
        threads = []
        for k in range(n_workers):
            thread = Thread(target=_worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # put data on the queue!
        for batch_pairs in iter_pairs:
            jobs.put(batch_pairs)

        # no more data, tell the threads to stop
        for i in range(len(threads)):
            jobs.put(None)

        # now join the threads
        for thread in threads:
            thread.join()

    def train_batch(self, np.ndarray[uint32_t, ndim=2, mode='c'] pairs):
        '''
        Update the model with a single batch of pairs
        '''
        with nogil:
            train_batch(&pairs[0, 0], pairs.shape[0],
                        self.energy_func, self.gradient_func,
                        self.mu_ptr, self.sigma_ptr, self.covariance_type,
                        self.N, self.K,
                        &self.eta, self.Closs,
                        self.mu_max, self.sigma_min, self.sigma_max,
                        self.acc_grad_mu_ptr, self.acc_grad_sigma_ptr
                        )

    def energy(self, i, j, func=None):
        '''
        Compute the energy between i and j

        This a wrapper around the cython code for convenience

        If func==None (default) then use self.energy_type, otherwise
            use the specified type
        '''
        cdef energy_t efunc
        if func is None:
            return self.energy_func(i, j, 2,
                                    self.mu_ptr, self.sigma_ptr, self.covariance_type,
                                    self.N, self.K)
        else:
            if func == 'IP':
                efunc = <energy_t> ip_energy
            elif func == 'KL':
                efunc = <energy_t> kl_energy
            else:
                raise ValueError
            return efunc(i, j, 2,
                         self.mu_ptr, self.sigma_ptr, self.covariance_type,
                         self.N, self.K)

    def gradient(self, i, j):
        '''
        Compute the gradient with i and j

        This a wrapper around the cython code for convenience
        Returns (dmui, dsigmai), (dmuj, dsigmaj)
        '''
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] dmui
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] dmuj
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] dsigmai
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] dsigmaj

        dmui = np.zeros(self.K, dtype=DTYPE)
        dmuj = np.zeros(self.K, dtype=DTYPE)
        if self.covariance_type == SPHERICAL:
            dsigmai = np.zeros(1, dtype=DTYPE)
            dsigmaj = np.zeros(1, dtype=DTYPE)
        elif self.covariance_type == DIAGONAL:
            dsigmai = np.zeros(self.K, dtype=DTYPE)
            dsigmaj = np.zeros(self.K, dtype=DTYPE)

        self.gradient_func(i, j, 2,
                           &dmui[0], &dsigmai[0], &dmuj[0], &dsigmaj[0],
                           self.mu_ptr, self.sigma_ptr, self.covariance_type,
                           self.N, self.K)
        return (dmui, dsigmai), (dmuj, dsigmaj)

cdef void _get_muij(size_t i, size_t j, size_t center_index,
                    size_t N, size_t K,
                    DTYPE_t*mu_ptr, DTYPE_t** mui_ptr, DTYPE_t** muj_ptr) nogil:
    # get pointers to mu[i] and mu[j] depending on the center_index
    # to modify the pointers we need to pass pointers to pointers
    # and use mu_ptr[0] instead of *mu_ptr

    if center_index == 0:
        mui_ptr[0] = mu_ptr + i * K
        muj_ptr[0] = mu_ptr + (j + N) * K
    elif center_index == 1:
        mui_ptr[0] = mu_ptr + (i + N) * K
        muj_ptr[0] = mu_ptr + j * K
    else:
        mui_ptr[0] = mu_ptr + i * K
        muj_ptr[0] = mu_ptr + j * K

cdef void _get_sigmaij(size_t i, size_t j, size_t center_index,
                       uint32_t covariance_type, size_t N, size_t K,
                       DTYPE_t*sigma_ptr, DTYPE_t** sigmai_ptr, DTYPE_t** sigmaj_ptr) nogil:
    cdef size_t fac

    if covariance_type == SPHERICAL:
        fac = 1
    elif covariance_type == DIAGONAL:
        fac = K

    if center_index == 0:
        sigmai_ptr[0] = sigma_ptr + i * fac
        sigmaj_ptr[0] = sigma_ptr + (j + N) * fac
    elif center_index == 1:
        sigmai_ptr[0] = sigma_ptr + (i + N) * fac
        sigmaj_ptr[0] = sigma_ptr + j * fac
    else:
        sigmai_ptr[0] = sigma_ptr + i * fac
        sigmaj_ptr[0] = sigma_ptr + j * fac

cdef DTYPE_t kl_energy(size_t i, size_t j, size_t center_index,
                       DTYPE_t*mu_ptr, DTYPE_t*sigma_ptr, uint32_t covariance_type,
                       size_t N, size_t K) nogil:
    '''
    Implementation of KL-divergence energy function
    '''
    #   E(P[i], P[j]) = -0.5 * (
    #       Tr((Sigma[i] ** -1) * Sigma[j]) +
    #       (mu[i] - mu[j]).T * (Sigma[i] ** -1) * (mu[i] - mu[j]) -
    #       d - log(det(Sigma[j]) / det(Sigma[i]))
    #   )

    cdef DTYPE_t det_fac
    cdef DTYPE_t trace_fac
    cdef DTYPE_t mu_diff_sq, mu_diff
    cdef DTYPE_t sigma_ratio
    cdef DTYPE_t*mui_ptr
    cdef DTYPE_t*muj_ptr
    cdef DTYPE_t*sigmai_ptr
    cdef DTYPE_t*sigmaj_ptr

    cdef size_t k

    _get_muij(i, j, center_index, N, K, mu_ptr, &mui_ptr, &muj_ptr)
    _get_sigmaij(i, j, center_index, covariance_type, N, K,
                 sigma_ptr, &sigmai_ptr, &sigmaj_ptr)

    if covariance_type == SPHERICAL:
        # log(det(sigma[j] / sigma[i]) = log det sigmaj - log det sigmai
        #   det = sigma ** K  SO
        # = K * (log(sigmaj) - log(sigmai)) = K * log(sigma j / sigmai)
        sigma_ratio = sigmaj_ptr[0] / sigmai_ptr[0]
        det_fac = K * log(sigma_ratio)
        trace_fac = K * sigma_ratio

        mu_diff_sq = 0.0
        for k in range(K):
            mu_diff = mui_ptr[k] - muj_ptr[k]
            mu_diff_sq += mu_diff * mu_diff

        return -0.5 * (
            trace_fac
            + mu_diff_sq / sigmai_ptr[0]
            - K - det_fac
        )

    elif covariance_type == DIAGONAL:
        # for det piece:
        # det sigmaj = prod (all entries on diagonal) so
        # log(det(Sigmaj) / det(Sigmai)) =
        # = log det Sigmaj - log det Sigmai
        # = log prod Sigmaj[k] - log prod Sigmai[k]
        # = SUM (log Sigmaj[k]) - SUM (log Sigmai[k)
        # = SUM (log Sigmaj[k] - log Sigmai[k])
        # = SUM (log (Sigmaj[k] / Sigmai[k]))
        # = log (prod Sigmaj[k] / Sigmai[k])

        trace_fac = 0.0
        mu_diff_sq = 0.0
        det_fac = 1.0
        for k in range(K):
            sigma_ratio = sigmaj_ptr[k] / sigmai_ptr[k]
            trace_fac += sigma_ratio
            mu_diff = mui_ptr[k] - muj_ptr[k]
            mu_diff_sq += mu_diff * mu_diff / sigmai_ptr[k]
            det_fac *= sigma_ratio

        # bound det_fac
        if det_fac < 1.0e-8:
            det_fac = 1.0e-8
        elif det_fac > 1.0e8:
            det_fac = 1.0e8

        return -0.5 * (trace_fac + mu_diff_sq - K - log(det_fac))

cdef void kl_gradient(size_t i, size_t j, size_t center_index,
                      DTYPE_t*dEdmui_ptr, DTYPE_t*dEdsigmai_ptr,
                      DTYPE_t*dEdmuj_ptr, DTYPE_t*dEdsigmaj_ptr,
                      DTYPE_t*mu_ptr, DTYPE_t*sigma_ptr, uint32_t covariance_type,
                      size_t N, size_t K) nogil:
    '''
    Implementation of KL-divergence gradient

    dEdmu and dEdsigma are used for return values.
    They must hold enough space for the particular covariance matrix
    (for spherical dEdsigma are floats, for diagonal are K dim arrays
    '''
    #   dE/dmu[i] = - dE/dmu[j] = -Delta'[i, j]
    #   dE/Sigma[i] = 0.5 * (
    #       (Sigma[i] ** -1) * Sigma[j] * (Sigma[i] ** -1) +
    #       Delta'[i, j] * Delta'[i, j].T -
    #       (Sigma[i] ** -1)
    #   )
    #   dE/dSigma[j] = 0.5 * ((Sigma[j] ** -1) - (Sigma[i] ** -1))
    #  Delta'[i, j] = (Sigma[i] ** -1) * (mu[i] - mu[j])
    cdef DTYPE_t deltaprime
    cdef DTYPE_t sum_deltaprime2
    cdef DTYPE_t si, sj, one_over_si, sj_si2
    cdef size_t k

    cdef DTYPE_t*mui_ptr
    cdef DTYPE_t*muj_ptr
    cdef DTYPE_t*sigmai_ptr
    cdef DTYPE_t*sigmaj_ptr

    _get_muij(i, j, center_index, N, K, mu_ptr, &mui_ptr, &muj_ptr)
    _get_sigmaij(i, j, center_index, covariance_type, N, K,
                 sigma_ptr, &sigmai_ptr, &sigmaj_ptr)

    if covariance_type == SPHERICAL:
        # compute deltaprime and assign it to dEdmu
        # Note:
        # Delta'[i, j] * Delta'[i, j].T is a dense K x K matrix, but we
        # only have one parameter that is tied along the diagonal.
        # so, use the diagnonal elements of the full matrix
        sum_deltaprime2 = 0.0
        one_over_si = 1.0 / sigmai_ptr[0]
        for k in range(K):
            deltaprime = one_over_si * (mui_ptr[k] - muj_ptr[k])
            dEdmui_ptr[k] = -deltaprime
            dEdmuj_ptr[k] = deltaprime
            sum_deltaprime2 += deltaprime * deltaprime

        dEdsigmai_ptr[0] = 0.5 * (
            K * sigma_ptr[j] * (1.0 / sigmai_ptr[0]) ** 2
            + sum_deltaprime2
            - (K / sigmai_ptr[0])
        )
        dEdsigmaj_ptr[0] = 0.5 * (1.0 / sigmaj_ptr[0] - 1.0 / sigmai_ptr[0]) * K

    elif covariance_type == DIAGONAL:
        for k in range(K):
            si = sigmai_ptr[k]
            sj = sigmaj_ptr[k]

            # compute deltaprime and assign it to dEdmu
            deltaprime = (1.0 / si) * (mui_ptr[k] - muj_ptr[k])
            dEdmui_ptr[k] = -deltaprime
            dEdmuj_ptr[k] = deltaprime

        # splitting the loop here and writting sj / si ** 2 as below
        # allows vectorization
        for k in range(K):
            si = sigmai_ptr[k]
            sj = sigmaj_ptr[k]
            deltaprime = dEdmuj_ptr[k]

            # writing sj / si ** 2 like this allows vectorization
            # since gcc was trying to substitute sj / si / si with
            # sj * pow(si, -2)
            sj_si2 = sj / si
            sj_si2 /= si

            # just use the diagonal elements of Delta'[i, j] * Delta'[i, j].T
            dEdsigmai_ptr[k] = 0.5 * (
                sj_si2
                + deltaprime * deltaprime
                - 1.0 / si
            )
            dEdsigmaj_ptr[k] = 0.5 * (1.0 / sj - 1.0 / si)

cdef DTYPE_t ip_energy(size_t i, size_t j, size_t center_index,
                       DTYPE_t*mu_ptr, DTYPE_t*sigma_ptr, uint32_t covariance_type,
                       size_t N, size_t K) nogil:
    '''
    Implementation of Inner product (symmetric) energy function

    Returns log(E(i, j))
    '''
    # E(P[i], P[j]) = N(0; mu[i] - mu[j], Sigma[i] + Sigma[j])
    # and use log(E) =
    #   -0.5 log det(sigma[i] + sigma[j])
    #   - 0.5 * (mu[i] - mu[j]) * (sigma[i] + sigma[j]) ** -1 * (mu[i]-mu[j])
    #   - K / 2 * log(2 * pi)

    cdef DTYPE_t log_2_pi = 1.8378770664093453
    cdef DTYPE_t det_fac
    cdef DTYPE_t mu_diff_sq, mu_diff
    cdef DTYPE_t sigmai_plus_sigmaj
    cdef size_t k

    cdef DTYPE_t*mui_ptr
    cdef DTYPE_t*muj_ptr
    cdef DTYPE_t*sigmai_ptr
    cdef DTYPE_t*sigmaj_ptr

    _get_muij(i, j, center_index, N, K, mu_ptr, &mui_ptr, &muj_ptr)
    _get_sigmaij(i, j, center_index, covariance_type, N, K,
                 sigma_ptr, &sigmai_ptr, &sigmaj_ptr)

    if covariance_type == SPHERICAL:
        # log(det(sigma[i] + sigma[j]))
        # = log ((sigma[i] + sigma[j]) ** K)
        # = K * log(sigmai + sigmaj)
        det_fac = K * log(sigmai_ptr[0] + sigmaj_ptr[0])

        mu_diff_sq = 0.0
        for k in range(K):
            mu_diff = mui_ptr[k] - muj_ptr[k]
            mu_diff_sq += mu_diff * mu_diff

        return -0.5 * (
            det_fac +
            mu_diff_sq / (sigmai_ptr[0] + sigmaj_ptr[0]) +
            K * log_2_pi
        )

    elif covariance_type == DIAGONAL:
        # log(det(sigma[i] + sigma[j]))
        # = log PROD (sigma[i] + sigma[j])
        # = SUM log (sigma[i] + sigma[j])
        # = log prod (sigma[i] + sigma[j])

        det_fac = 1.0
        mu_diff_sq = 0.0

        for k in range(K):
            sigmai_plus_sigmaj = sigmai_ptr[k] + sigmaj_ptr[k]
            det_fac *= sigmai_plus_sigmaj
            mu_diff = mui_ptr[k] - muj_ptr[k]
            mu_diff_sq += mu_diff * mu_diff / sigmai_plus_sigmaj

        if det_fac < 1.0e-8:
            det_fac = 1.0e-8
        elif det_fac > 1.0e8:
            det_fac = 1.0e8

        return -0.5 * (log(det_fac) + mu_diff_sq + K * log_2_pi)

cdef void ip_gradient(size_t i, size_t j, size_t center_index,
                      DTYPE_t*dEdmui_ptr, DTYPE_t*dEdsigmai_ptr,
                      DTYPE_t*dEdmuj_ptr, DTYPE_t*dEdsigmaj_ptr,
                      DTYPE_t*mu_ptr, DTYPE_t*sigma_ptr, uint32_t covariance_type,
                      size_t N, size_t K) nogil:
    '''
    Implementation of Inner product based gradient
    '''
    #   dE/dmu[i] = - dE/dmu[j] = -Delta[i, j]
    #   dE/dSigma[i] = dE/dSigma[j] = 0.5 * (
    #   Delta[i, j] * Delta[i, j].T - (Sigma[i] + Sigma[j]) ** -1)
    #   Delta[i, j] = ((Sigma[i] + Sigma[j]) ** -1) * (mu[i] - mu[j])
    cdef DTYPE_t delta, sigmai_plus_sigmaj, sigmai_plus_sigmaj_inv, sum_delta2
    cdef DTYPE_t dEdsigma
    cdef size_t k

    cdef DTYPE_t*mui_ptr
    cdef DTYPE_t*muj_ptr
    cdef DTYPE_t*sigmai_ptr
    cdef DTYPE_t*sigmaj_ptr

    _get_muij(i, j, center_index, N, K, mu_ptr, &mui_ptr, &muj_ptr)
    _get_sigmaij(i, j, center_index, covariance_type, N, K,
                 sigma_ptr, &sigmai_ptr, &sigmaj_ptr)

    if covariance_type == SPHERICAL:
        # compute delta and assign it to dEdmu
        sigmai_plus_sigmaj_inv = 1.0 / (sigmai_ptr[0] + sigmaj_ptr[0])
        # we'll sum up delta ** 2 too
        sum_delta2 = 0.0
        for k in range(K):
            delta = sigmai_plus_sigmaj_inv * (mui_ptr[k] - muj_ptr[k])
            dEdmui_ptr[k] = -delta
            dEdmuj_ptr[k] = delta
            sum_delta2 += delta * delta

        dEdsigmai_ptr[0] = 0.5 * (
            + sum_delta2
            - sigmai_plus_sigmaj_inv * K
        )
        dEdsigmaj_ptr[0] = dEdsigmai_ptr[0]

    elif covariance_type == DIAGONAL:

        for k in range(K):
            sigmai_plus_sigmaj = sigmai_ptr[k] + sigmaj_ptr[k]
            sigmai_plus_sigmaj_inv = 1.0 / sigmai_plus_sigmaj
            delta = sigmai_plus_sigmaj_inv * (mui_ptr[k] - muj_ptr[k])

            dEdmui_ptr[k] = -delta
            dEdmuj_ptr[k] = delta

        # this break allows auto vectorization
        for k in range(K):
            sigmai_plus_sigmaj = sigmai_ptr[k] + sigmaj_ptr[k]
            sigmai_plus_sigmaj_inv = 1.0 / sigmai_plus_sigmaj
            delta = dEdmuj_ptr[k]

            dEdsigma = 0.5 * (delta * delta - sigmai_plus_sigmaj_inv)
            dEdsigmai_ptr[k] = dEdsigma
            dEdsigmaj_ptr[k] = dEdsigma

cdef void train_batch(
        uint32_t*pairs, size_t Npairs,
        energy_t energy_func, gradient_t gradient_func,
        DTYPE_t*mu_ptr, DTYPE_t*sigma_ptr, uint32_t covariance_type,
        size_t N, size_t K,
        LearningRates*eta, DTYPE_t Closs, DTYPE_t C, DTYPE_t m, DTYPE_t M,
        DTYPE_t*acc_grad_mu, DTYPE_t*acc_grad_sigma
) nogil:
    '''
    Update the model on a batch of data

    pairs = numpy array of positive / negative examples
        (see documentation of text_to_pairs for a detailed description)
    Npairs = number of training examples in this set
    '''
    cdef size_t k, posi, posj, negi, negj, pos_neg, i, j, center_index
    cdef DTYPE_t loss
    cdef DTYPE_t pos_energy, neg_energy
    cdef DTYPE_t fac

    # working space for the gradient
    # make one vector of length 4 * K, then partition it up for
    # the four different types of gradients
    # in the spherical case dsigmai, j need to be size 1, but size K
    # in diagonal case.  To save code complexity we'll always allocate as
    # size K since K is small
    cdef DTYPE_t *work = <DTYPE_t*> malloc(K * 4 * sizeof(DTYPE_t))
    cdef DTYPE_t *dmui = work
    cdef DTYPE_t *dmuj = work + K
    cdef DTYPE_t *dsigmai = work + 2 * K
    cdef DTYPE_t *dsigmaj = work + 3 * K

    for k in range(Npairs):

        # compute the loss
        # loss = max(0.0, Closs - energy(pos) + energy(neg))
        posi = pairs[k * 5]
        posj = pairs[k * 5 + 1]
        negi = pairs[k * 5 + 2]
        negj = pairs[k * 5 + 3]
        center_index = pairs[k * 5 + 4]

        pos_energy = energy_func(posi, posj, center_index,
                                 mu_ptr, sigma_ptr, covariance_type, N, K)
        neg_energy = energy_func(negi, negj, center_index,
                                 mu_ptr, sigma_ptr, covariance_type, N, K)
        loss = Closs - pos_energy + neg_energy

        if loss < 1.0e-14:
            # loss for this sample is 0, nothing to update
            continue

        # compute gradients and update
        # have almost identical calculations for postive and negative
        # except the sign of update
        for pos_neg in range(2):
            # this is a trick to get the right indices
            i = pairs[k * 5 + pos_neg * 2]
            j = pairs[k * 5 + pos_neg * 2 + 1]
            if pos_neg == 0:
                fac = -1.0
            else:
                fac = 1.0

            # compute the gradients
            gradient_func(i, j, center_index,
                          dmui, dsigmai, dmuj, dsigmaj,
                          mu_ptr, sigma_ptr, covariance_type, N, K)

            # accumulate the gradients for adagrad and update
            # parameters. center_index determines whether we update
            # the context parameters or center word parameters.
            # can handle both cases by appropriately modifying
            # the i, j passed to _accumulate_update.
            # if center_index == 0 then pass i and j + N
            # if center_index == 1 then pass i + N and j
            # so i + center_index * N and
            #    j + (1 - center_index) * N handles both cases
            _accumulate_update(i + center_index * N, dmui, dsigmai,
                               mu_ptr, sigma_ptr, covariance_type,
                               fac, eta, C, m, M, acc_grad_mu, acc_grad_sigma,
                               N, K)
            _accumulate_update(j + (1 - center_index) * N, dmuj, dsigmaj,
                               mu_ptr, sigma_ptr, covariance_type,
                               fac, eta, C, m, M, acc_grad_mu, acc_grad_sigma,
                               N, K)

    free(work)

cdef void _accumulate_update(
        size_t k, DTYPE_t* dmu, DTYPE_t* dsigma,
        DTYPE_t* mu_ptr, DTYPE_t* sigma_ptr, uint32_t covariance_type,
        DTYPE_t fac, LearningRates*eta, DTYPE_t C, DTYPE_t m, DTYPE_t M,
        DTYPE_t* acc_grad_mu, DTYPE_t* acc_grad_sigma,
        size_t N, size_t K
) nogil:
    # accumulate the gradients and update
    cdef size_t i
    cdef DTYPE_t local_eta, global_eta, eta_min
    cdef DTYPE_t l2_mu
    cdef DTYPE_t sig

    cdef DTYPE_t max_grad = 10.0

    # update for mu
    l2_mu = 0.0
    global_eta = eta.mu  # pre-assigning these allows gcc autovectorization
    eta_min = eta.mu_min
    for i in range(K):
        # clip gradients
        dmu[i] = (max_grad if dmu[i] > max_grad else dmu[i])
        dmu[i] = (-max_grad if dmu[i] < -max_grad else dmu[i])
        # update the accumulated gradient for adagrad
        acc_grad_mu[k * K + i] += dmu[i] * dmu[i]
        # now get local learning rate for this word
        local_eta = global_eta / (sqrt(acc_grad_mu[k * K + i]) + 1.0)
        local_eta = (eta_min if local_eta < eta_min else local_eta)
        # finally update mu
        mu_ptr[k * K + i] -= fac * local_eta * dmu[i]
        # accumulate L2 norm of mu for regularization
        l2_mu += mu_ptr[k * K + i] * mu_ptr[k * K + i]
    l2_mu = sqrt(l2_mu)

    # regularizer
    if l2_mu > C:
        l2_mu = C / l2_mu
        for i in range(K):
            mu_ptr[k * K + i] *= l2_mu

    # now for Sigma
    if covariance_type == SPHERICAL:
        dsigma[0] = (max_grad if dsigma[0] > max_grad else dsigma[0])
        dsigma[0] = (-max_grad if dsigma[0] < -max_grad else dsigma[0])
        acc_grad_sigma[k] += dsigma[0] * dsigma[0]
        local_eta = eta.sigma / (sqrt(acc_grad_sigma[k]) + 1.0)
        local_eta = (eta.sigma_min if local_eta < eta.sigma_min else local_eta)
        sigma_ptr[k] -= fac * local_eta * dsigma[0]
        if sigma_ptr[k] > M:
            sigma_ptr[k] = M
        elif sigma_ptr[k] < m:
            sigma_ptr[k] = m

    elif covariance_type == DIAGONAL:
        global_eta = eta.sigma
        eta_min = eta.sigma_min
        for i in range(K):
            # clip gradients
            dsigma[i] = (max_grad if dsigma[i] > max_grad else dsigma[i])
            dsigma[i] = (-max_grad if dsigma[i] < -max_grad else dsigma[i])
            # update the accumulated gradient for adagrad
            acc_grad_sigma[k * K + i] += dsigma[i] * dsigma[i]
            # now get local learning rate for this word
            local_eta = global_eta / (sqrt(acc_grad_sigma[k * K + i]) + 1.0)
            local_eta = (eta_min if local_eta < eta_min else local_eta)
            # finally update sigma
            sigma_ptr[k * K + i] -= fac * local_eta * dsigma[i]
            # bound sigma between m and M
            # note: the ternary operator instead of if statment
            #   allows cython to generate code that gcc will vectorize
            sigma_ptr[k * K + i] = (M if sigma_ptr[k * K + i] > M
                                    else sigma_ptr[k * K + i])
            sigma_ptr[k * K + i] = (m if sigma_ptr[k * K + i] < m
                                    else sigma_ptr[k * K + i])

cpdef np.ndarray[uint32_t, ndim=2, mode='c'] text_to_pairs(
        text, random_gen, uint32_t half_window_size=2,
        uint32_t nsamples_per_word=1):
    '''
    Take a chunk of text and turn it into a array of pairs for training.

    The return array pairs is size (npairs, 5).
    Each row is one training example.  The first two entries are the
    token ids for the positive example and the second two entries
    are the token ids for the negative example.  The final entry
    is either 0 or 1 with encoding whether the left (0) or right (1)
    id is the center word.  For example, this row:
        [3, 122, 3, 910, 0]
    means: word id's 3 and 122 occur in a context window, word ids 3 and 910
    are the negative sample and the 0 signals word 3 is the center word.

    text is a list of text documents / sentences.

    Each element of the list is a numpy array of uint32_t IDs, with UINT32_MAX
    signifying an OOV ID representing the document or sentence.

    For position k in the document, uses all contexts from k - half_window_size
    to k + half_window_size

    random_gen = a callable that returns random IDs:
        array of uint32_t length N with random IDs = random_gen(N)
    nsamples_per_words = for each positive pair, sample this many negative
        pairs
    '''
    # calculate number of windows we need
    # for each token, take all positive indices in half_window_size
    # and take two samples from it (one for left token, one for right token)
    # for each nsamples_per_word.  Note that this includes full windows
    # for words at end, so it slightly overestimates number of pairs
    cdef long long npairs = sum(
        [2 * len(doc) * half_window_size * nsamples_per_word for doc in text]
    )

    # allocate pairs and draw random numbers
    cdef np.ndarray[uint32_t, ndim=2, mode='c'] pairs = np.empty(
        (npairs, 5), dtype=np.uint32)
    cdef np.ndarray[uint32_t] randids = random_gen(npairs)
    cdef np.ndarray[uint32_t, ndim=1, mode='c'] cdoc

    cdef size_t next_pair = 0  # index of next pair to write
    cdef size_t i, j, k
    cdef uint32_t doc_len

    for doc in text:
        cdoc = doc
        doc_len = cdoc.shape[0]
        for i in range(doc_len):
            if cdoc[i] == UINT32_MAX:
                # OOV word
                continue
            for j in range(i + 1, min(i + half_window_size + 1, doc_len)):
                if cdoc[j] == UINT32_MAX:
                    # OOV word
                    continue
                # take nsamples_per_word samples
                # we ignore handling the case where sample is i or j, for now
                for k in range(nsamples_per_word):
                    # sample j
                    pairs[next_pair, 0] = cdoc[i]
                    pairs[next_pair, 1] = cdoc[j]
                    pairs[next_pair, 2] = cdoc[i]
                    pairs[next_pair, 3] = randids[next_pair]
                    pairs[next_pair, 4] = 0
                    next_pair += 1

                    # now sample i
                    pairs[next_pair, 0] = cdoc[i]
                    pairs[next_pair, 1] = cdoc[j]
                    pairs[next_pair, 2] = randids[next_pair]
                    pairs[next_pair, 3] = cdoc[j]
                    pairs[next_pair, 4] = 1
                    next_pair += 1

    return np.ascontiguousarray(pairs[:next_pair, :])

# a wrapper function for unicode strings
cdef unicode _ustring(s):
    if type(s) is unicode:
        return <unicode> s
    elif isinstance(s, bytes):
        return (<bytes> s).decode('ascii')
    elif isinstance(s, unicode):
        return unicode(s)
    else:
        raise TypeError('expected unicode, got {}'.format(type(s)))
