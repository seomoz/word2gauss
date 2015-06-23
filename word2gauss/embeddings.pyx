#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True


# have words/tokens w[i], i = 1 .. N
# Each word has probability P[i] = gaussian
# The Gaussian is parameterized by
#
#   P ~ N(x; mu[i], Sigma[i])
#
# where mu[i] = vector length K
# Sigma[i] = covariance matrix = (K, K) array
#
# Make two approximations to simply Sigma[i]:
#
# Either it's 'diagonal' in which case Sigma[i] can be represented by
#   sigma[i] = vector length K
# OR 'spherical' = diagonal with every element the same when Sigma[i] = float
#
#
# Now define Energy function E(P[i], P[j]) = similarity like measure of
# the two probabilities.
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

#  for each loss need:
#  functions to compute E(i, j) and dE/dparam
#  This depends on the parameterization for sigma so two parameterizations,
#  two losses = 4 X code (bad)

# Model class:
# map word -> id
# numpy array (nwords, ndim of vector) for means
# covariance: numpy array (nwords, ndim of vector) for diagonal
#                         (nwords, 1) for spherical
# Break off means / covariance into a separate Distribution class
# Bundle with vocab / tokenizer / etc

# Distribution class:
#  holds arrays and covariances of different types
#  also holds the model parameters
# public methods:
#   .covariance_type = '...' attribute
#   .mu = numpy array of mus
#   .Sigma = appropriately sized array for Sigma
#  probablility of a given vector
#
# Model class holds a distribution class + vocab + tokenizer, etc
#
# To train Distribution:
# initialize with size of vocab and random weights
# has method:
# def train(self, iterator of postive and negative pairs, e.g.
#   iterator of [(i_pos, j_pos), (i_neg, j_neg)])

# Inside train:
# Split pairs into minibatches
# In each batch:
#   iterate through pairs
#       compute gradients
#       get learning rate, update global parameters, update learning rate

# read through sentences:
#   split into batches
#   in each batch:
#       for each sentence
#           read through words and accumulate gradient
#       update global parameters with accumulated gradient
#
# NOTE: C/C++ is row major (so A[i, :] is contiguous)
#

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

LOGGER = logging.getLogger()


cdef extern from "stdint.h":
    ctypedef unsigned long long uint32_t

# type of our floating point arrays
cdef type DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# types of covariance matrices we support
cdef uint32_t SPHERICAL = 1
#cdef uint32 DIAGONAL = 2 # maybe add diagonal in the future?
COV_MAP = {1: 'spherical', 2: 'diagonal'}

# define a generic interface for energy and gradient functions
# (e.g. symmetric or KL-divergence).  This defines a type for the function...

# interface for an energy function
# energy(i, j, # word indices to compute energy for
#        mu_ptr,  # pointer to mu array
#        sigma_ptr,  # pointer to sigma array
#        covariance_type (SPHERICAL, etc)
#        nwords, ndimensions)
ctypedef DTYPE_t (*energy_t)(size_t, size_t,
    DTYPE_t*, DTYPE_t*, uint32_t, size_t, size_t) nogil

# interface for a gradient function
# gradient(i, j, # word indices to compute energy for
#          dEdmui_ptr, dEdsigma1_ptr, dEdmuj_ptr, dEdsigmaj_ptr
#          mu_ptr,  # pointer to mu array
#          sigma_ptr,  # pointer to sigma array
#          covariance_type (SPHERICAL, etc)
#          nwords, ndimensions)
ctypedef void (*gradient_t)(size_t, size_t,
    DTYPE_t*, DTYPE_t*, DTYPE_t*, DTYPE_t*,
    DTYPE_t*, DTYPE_t*, uint32_t, size_t, size_t) nogil


cdef class GaussianEmbedding:

    cdef np.ndarray mu, sigma, acc_grad_mu, acc_grad_sigma
    cdef uint32_t covariance_type
    cdef string energy_type

    # number of words, dimension of vectors
    cdef size_t N
    cdef size_t K

    # L2 regularization parameters
    cdef DTYPE_t mu_max, sigma_min, sigma_max

    # training parameters
    cdef DTYPE_t eta, Closs

    # energy and gradient functions
    cdef energy_t energy_func
    cdef gradient_t gradient_func

    # pointers to data arrays
    cdef DTYPE_t* mu_ptr
    cdef DTYPE_t* sigma_ptr
    cdef DTYPE_t* acc_grad_mu_ptr
    cdef DTYPE_t* acc_grad_sigma_ptr

    def __cinit__(self, N, size=100,
        covariance_type='spherical', mu_max=1.0, sigma_min=0.1, sigma_max=1.0,
        energy_type='KL',
        init_params={
            'mu0': 0.1,
            'sigma_mean0': 0.5,
            'sigma_std0': 1.0
        },
        eta=1.0, Closs=1.0,
        mu=None, sigma=None):
        '''
        N = number of distributions (e.g. number of words)
        size = dimension of each Gaussian
        covariance_type = 'spherical' or ...
        energy_type = 'KL' or 'IP'
        mu_max = maximum L2 norm of each mu
        sigma_min, sigma_max = maximum/min eigenvalues of sigma
        init_params = {
            mu0: initial means are random normal w/ mean=0, std=mu0
            sigma_mean0, sigma_std0: initial sigma is random normal with
                mean=sigma_mean0, std=sigma_std0
            NOTE: these are ignored if mu/sigma is explicitly specified
        }
        eta = global learning rate
        Closs = regularization parameter in max-margin loss
            loss = max(0.0, Closs - energy(pos) + energy(neg)
        if mu/sigma are not None then they specify the initial values
            for the parameters
        '''
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _sigma
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] _acc_grad_mu
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] _acc_grad_sigma

        if covariance_type == 'spherical':
            self.covariance_type = SPHERICAL
        else:
            raise ValueError
    
        self.energy_type = energy_type
        if energy_type == 'KL':
            self.energy_func = <energy_t>kl_energy
            self.gradient_func = <gradient_t>kl_gradient
        elif energy_type == 'IP':
            self.energy_func = <energy_t>ip_energy
        else:
            raise ValueError

        self.N = N 
        self.K = size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu_max = mu_max
        self.eta = eta
        self.Closs = Closs

        # Initialize parameters 
        if mu is None:
            _mu = init_params['mu0'] * np.ascontiguousarray(
                np.random.randn(self.N, self.K).astype(DTYPE))
        else:
            assert mu.shape[0] == N
            assert mu.shape[1] == self.K
            _mu = mu
        self.mu = _mu

        if sigma is None:
            if self.covariance_type == SPHERICAL:
                s = np.random.randn(self.N, 1).astype(DTYPE)
            s *= init_params['sigma_std0']
            s += init_params['sigma_mean0']
            _sigma = np.ascontiguousarray(
                np.maximum(sigma_min, np.minimum(s, sigma_max)))
        else:
            assert sigma.shape[0] == N
            if self.covariance_type == SPHERICAL:
                assert sigma.shape[1] == 1
            _sigma = sigma
        self.sigma = _sigma

        assert _mu.flags['C_CONTIGUOUS']
        assert _sigma.flags['C_CONTIGUOUS']
        self.mu_ptr = &_mu[0, 0]
        self.sigma_ptr = &_sigma[0, 0]

        # working space for accumulated gradients
        _acc_grad_mu = np.ascontiguousarray(np.zeros(N, dtype=DTYPE))
        assert _acc_grad_mu.flags['C_CONTIGUOUS']
        self.acc_grad_mu = _acc_grad_mu
        self.acc_grad_mu_ptr = &_acc_grad_mu[0]

        if self.covariance_type == SPHERICAL:
            _acc_grad_sigma = np.ascontiguousarray(np.zeros(N, dtype=DTYPE))
        assert _acc_grad_sigma.flags['C_CONTIGUOUS']
        self.acc_grad_sigma = _acc_grad_sigma
        self.acc_grad_sigma_ptr = &_acc_grad_sigma[0]

    def save(self, fname, vocab=None, full=True):
        '''
        Writes a gzipped text file of the model in word2vec text format

        word_or_id1 0.5 -0.2 1.0 ...
        word_or_id2 ...

        This class doesn't have knowledge of id -> word mapping, so
        it can be passed in as a callable vocab(id) return the word string

        if full=True, then writes out the full model.  It is a tar.gz
        file the word vectors and Sigma files, in addition
        to files for the other state (accumulated gradient for
            training, model parameters)

        It has files:
            word_mu: the word2vec file with word/id and mu vectors
            sigma: the sigma for each vector, one per line
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
            for i in xrange(self.N):
                line = [vocab(i)] + self.mu[i, :].tolist()
                fout.write(' '.join('%s' % ele for ele in line) + '\n')

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
            with NamedTemporaryFile() as tmp:
                tmp.write(json.dumps(params))
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
                params = json.loads(f.read())

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

        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _sigma
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] _acc_grad_mu
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] _acc_grad_sigma

        # set the data
        with open_tar(fname, 'r') as fin:
            _mu = np.empty((N, K), dtype=DTYPE)
            with closing(fin.extractfile('word_mu')) as f:
                for i, line in enumerate(f):
                    ls = line.strip().split()
                    # ls[0] is the word/id, skip it.  rest are mu
                    _mu[i, :] = [float(ele) for ele in ls[1:]]

            with closing(fin.extractfile('sigma')) as f:
                _sigma = np.loadtxt(f, dtype=DTYPE).reshape(N, -1).copy()
            with closing(fin.extractfile('acc_grad_mu')) as f:    
                _acc_grad_mu = np.loadtxt(f, dtype=DTYPE).reshape(N, ).copy()
            with closing(fin.extractfile('acc_grad_sigma')) as f:
                _acc_grad_sigma = np.loadtxt(f, dtype=DTYPE).reshape(N, ).copy()

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
        self.acc_grad_mu_ptr = &_acc_grad_mu[0]

        self.acc_grad_sigma = _acc_grad_sigma
        assert (
            self.acc_grad_sigma.flags['C_CONTIGUOUS'] and 
            self.acc_grad_sigma.flags['OWNDATA']
        )
        self.acc_grad_sigma_ptr = &_acc_grad_sigma[0]

    def nearest_neighbors(self, word_id, metric=cosine, num=10):
        '''Return the num nearest neighbors to word_id, using the metric

        Metric is either 'IP', 'KL' or is callable with this interface:
            array(N) with similarities = metric(
                array(N, K) of all vectors, array(K) of word_id
            )
        Default is cosine similarity.

        Returns (top num ids, top num scores)
        '''
        if metric == 'IP':
            scores = np.zeros(self.N)
            for k in xrange(self.N):
                scores[k] = self.energy(k, word_id, func=metric)
        elif metric == 'KL':
            scores = np.zeros(self.N)
            for k in xrange(self.N):
                scores[k] = 0.5 * (
                    self.energy(k, word_id, func=metric)
                    + self.energy(word_id, k, func=metric)
                )
        else:
            scores = metric(self.mu, self.mu[word_id, :].reshape(1, -1))
        sorted_indices = scores.argsort()[::-1][:(num + 1)]
        top_indices = [ele for ele in sorted_indices if ele != word_id]
        return top_indices, scores[top_indices]

    def __getattr__(self, name):
        if name == 'mu':
            return self.mu
        elif name == 'sigma':
            return self.sigma
        elif name == 'acc_grad_mu':
            return self.acc_grad_mu
        elif name == 'acc_grad_sigma':
            return self.acc_grad_sigma
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
        from Queue import Queue
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
        for k in xrange(n_workers):
            thread = Thread(target=_worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # put data on the queue!
        for batch_pairs in iter_pairs:
            jobs.put(batch_pairs)

        # no more data, tell the threads to stop
        for i in xrange(len(threads)):
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
                self.eta, self.Closs,
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
            return self.energy_func(i, j,
                self.mu_ptr, self.sigma_ptr, self.covariance_type,
                self.N, self.K)
        else:
            if func == 'IP':
                efunc = <energy_t>ip_energy
            elif func == 'KL':
                efunc = <energy_t>kl_energy
            else:
                raise ValueError
            return efunc(i, j,
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

        self.gradient_func(i, j,
            &dmui[0], &dsigmai[0], &dmuj[0], &dsigmaj[0],
            self.mu_ptr, self.sigma_ptr, self.covariance_type,
            self.N, self.K)
        return (dmui, dsigmai), (dmuj, dsigmaj)


cdef DTYPE_t kl_energy(size_t i, size_t j,
    DTYPE_t* mu_ptr, DTYPE_t* sigma_ptr, uint32_t covariance_type,
    size_t N, size_t K) nogil:
    '''
    Implementation of KL-divergence energy function
    '''
    #   E(P[i], P[j]) = 0.5 * (
    #       Tr((Sigma[i] ** -1) * Sigma[j]) +
    #       (mu[i] - mu[j]).T * (Sigma[i] ** -1) * (mu[i] - mu[j]) -
    #       d - log(det(Sigma[j]) / det(Sigma[i]))
    #   )

    cdef DTYPE_t det_fac
    cdef DTYPE_t trace_fac

    cdef DTYPE_t mu_diff_sq
    cdef size_t k

    if covariance_type == SPHERICAL:
        # log(det(sigma[j] / sigma[i]) = log det sigmaj - log det sigmai
        #   det = sigma ** K  SO
        # = K * (log(sigmaj) - log(sigmai)) = K * log(sigma j / sigmai)
        det_fac = K * log(sigma_ptr[j] / sigma_ptr[i])
        trace_fac = K * sigma_ptr[j] / sigma_ptr[i]

        mu_diff_sq = 0.0
        for k in range(K):
            mu_diff_sq += (mu_ptr[i * K + k] - mu_ptr[j * K + k]) ** 2

        return -0.5 * (
            trace_fac
            + mu_diff_sq / sigma_ptr[i]
            - K - det_fac
        )

cdef void kl_gradient(size_t i, size_t j,
    DTYPE_t* dEdmui_ptr, DTYPE_t* dEdsigmai_ptr,
    DTYPE_t* dEdmuj_ptr, DTYPE_t* dEdsigmaj_ptr,
    DTYPE_t* mu_ptr, DTYPE_t* sigma_ptr, uint32_t covariance_type,
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
    cdef size_t k

    if covariance_type == SPHERICAL:
        # compute deltaprime and assign it to dEdmu
        sum_deltaprime2 = 0.0
        for k in xrange(K):
            deltaprime = (1.0 / sigma_ptr[i]) * (
                mu_ptr[i * K + k] - mu_ptr[j * K + k])
            dEdmui_ptr[k] = -deltaprime
            dEdmuj_ptr[k] = deltaprime
            sum_deltaprime2 += deltaprime ** 2

        dEdsigmai_ptr[0] = 0.5 * (
            sigma_ptr[j] * (1.0 / sigma_ptr[i]) ** 2
            + sum_deltaprime2
            - (1.0 / sigma_ptr[i])
        )
        dEdsigmaj_ptr[0] = 0.5 * (1.0 / sigma_ptr[j] - 1.0 / sigma_ptr[i])


cdef DTYPE_t ip_energy(size_t i, size_t j,
    DTYPE_t* mu_ptr, DTYPE_t* sigma_ptr, uint32_t covariance_type,
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
    cdef DTYPE_t mu_diff_sq
    cdef size_t k

    if covariance_type == SPHERICAL:
        # log(det(sigma[i] + sigma[j]))
        # = log ((sigma[i] + sigma[j]) ** K)
        # = K * log(sigmai + sigmaj)
        det_fac = K * log(sigma_ptr[i] + sigma_ptr[j])

        mu_diff_sq = 0.0
        for k in range(K):
            mu_diff_sq += (mu_ptr[i * K + k] - mu_ptr[j * K + k]) ** 2

        return -0.5 * (
            det_fac +
            mu_diff_sq / (sigma_ptr[i] + sigma_ptr[j]) +
            K * log_2_pi
        )


cdef void train_batch(
    uint32_t* pairs, size_t Npairs,
    energy_t energy_func, gradient_t gradient_func,
    DTYPE_t* mu_ptr, DTYPE_t* sigma_ptr, uint32_t covariance_type,
    size_t N, size_t K,
    DTYPE_t eta, DTYPE_t Closs, DTYPE_t C, DTYPE_t m, DTYPE_t M,
    DTYPE_t* acc_grad_mu, DTYPE_t* acc_grad_sigma
    ) nogil:
    '''
    Update the model on a batch of data

    pairs = numpy array of positive negative pair = (Npairs, 4)
        row[0:2] is the positive i, j indices and row[2:4] are the
        negative indices
    Npairs = number of training examples in this set
    '''
    cdef size_t k, posi, posj, negi, negj, pos_neg, i, j
    cdef DTYPE_t loss
    cdef DTYPE_t pos_energy, neg_energy
    cdef DTYPE_t fac

    # working space for the gradient
    # make one vector of length 4 * K, then partition it up for
    # the four different types of gradients
    cdef DTYPE_t* work = <DTYPE_t*>malloc(K * 4 * sizeof(DTYPE_t))
    cdef DTYPE_t* dmui = work
    cdef DTYPE_t* dmuj = work + K
    cdef DTYPE_t* dsigmai = work + 2 * K
    cdef DTYPE_t* dsigmaj = work + 3 * K

    for k in range(Npairs):

        # compute the loss
        # loss = max(0.0, Closs - energy(pos) + energy(neg))
        posi = pairs[k * 4]
        posj = pairs[k * 4 + 1]
        negi = pairs[k * 4 + 2]
        negj = pairs[k * 4 + 3]

        pos_energy = energy_func(posi, posj,
            mu_ptr, sigma_ptr, covariance_type, N, K)
        neg_energy = energy_func(negi, negj, 
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
            i = pairs[k * 4 + pos_neg * 2]
            j = pairs[k * 4 + pos_neg * 2 + 1]
            if pos_neg == 0:
                fac = -1.0
            else:
                fac = 1.0

            # compute the gradients
            gradient_func(i, j,
                dmui, dsigmai, dmuj, dsigmaj,
                mu_ptr, sigma_ptr, covariance_type, N, K)

            _accumulate_update(i, dmui, dsigmai,
                mu_ptr, sigma_ptr, covariance_type,
                fac, eta, C, m, M, acc_grad_mu, acc_grad_sigma,
                N, K)
            _accumulate_update(j, dmuj, dsigmaj,
                mu_ptr, sigma_ptr, covariance_type,
                fac, eta, C, m, M, acc_grad_mu, acc_grad_sigma,
                N, K)

    free(work)

cdef void _accumulate_update(
    size_t k, DTYPE_t* dmu, DTYPE_t* dsigma,
    DTYPE_t* mu_ptr, DTYPE_t* sigma_ptr, uint32_t covariance_type,
    DTYPE_t fac, DTYPE_t eta, DTYPE_t C, DTYPE_t m, DTYPE_t M,
    DTYPE_t* acc_grad_mu, DTYPE_t* acc_grad_sigma,
    size_t N, size_t K
    ) nogil:

    # accumulate the gradients and update
    cdef size_t i
    cdef DTYPE_t sum_dmu2
    cdef DTYPE_t local_eta
    cdef DTYPE_t l2_mu

    # update for mu
    # first update the accumulated gradient for adagrad
    sum_dmu2 = 0.0
    for i in range(K):
        sum_dmu2 += dmu[i] ** 2
    sum_dmu2 /= K
    acc_grad_mu[k] += sum_dmu2

    # now get local learning rate for this word
    local_eta = eta / (sqrt(acc_grad_mu[k]) + 1.0)

    # finally update mu
    l2_mu = 0.0
    for i in range(K):
        mu_ptr[k * K + i] -= fac * local_eta * dmu[i]
        l2_mu += mu_ptr[k * K + i] ** 2
    l2_mu = sqrt(l2_mu)

    # regularizer
    if l2_mu > C:
        l2_mu = C / l2_mu
        for i in range(K):
            mu_ptr[k * K + i] *= l2_mu
    
    # now for Sigma
    if covariance_type == SPHERICAL:
        acc_grad_sigma[k] += dsigma[0] ** 2
        local_eta = eta / (sqrt(acc_grad_sigma[k]) + 1.0)
        sigma_ptr[k] -= fac * local_eta * dsigma[0]
        if sigma_ptr[k] > M:
            sigma_ptr[k] = M
        elif sigma_ptr[k] < m:
            sigma_ptr[k] = m

cpdef np.ndarray[uint32_t, ndim=2, mode='c'] text_to_pairs(
    text, random_gen, uint32_t half_window_size=2,
    uint32_t nsamples_per_word=1):
    '''
    Take a chunk of text and turn it into a array of pairs for training.

    text is a list of text documents / sentences.

    Each element of the list is a numpy array of uint32_t IDs, with -1
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
        (npairs, 4), dtype=np.uint32)
    cdef np.ndarray[uint32_t] randids = random_gen(npairs)
    cdef np.ndarray[uint32_t, ndim=1, mode='c'] cdoc

    cdef size_t next_pair = 0  # index of next pair to write
    cdef size_t i, j, k
    cdef uint32_t doc_len

    for doc in text:
        cdoc = doc
        doc_len = cdoc.shape[0]
        for i in range(doc_len):
            if cdoc[i] == -1:
                # OOV word
                continue
            for j in range(i + 1, min(i + half_window_size + 1, doc_len)):
                if cdoc[j] == -1:
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
                    next_pair += 1

                    # now sample i
                    pairs[next_pair, 0] = cdoc[i]
                    pairs[next_pair, 1] = cdoc[j]
                    pairs[next_pair, 2] = randids[next_pair]
                    pairs[next_pair, 3] = cdoc[j]
                    next_pair += 1

    return np.ascontiguousarray(pairs[:next_pair, :])

