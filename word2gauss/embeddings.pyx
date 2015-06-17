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

import numpy as np

from .utils import cosine


cdef extern from "stdint.h":
    ctypedef unsigned long long uint32_t

# type of our floating point arrays
cdef type DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# types of covariance matrices we support
cdef uint32_t SPHERICAL = 1
#cdef uint32 DIAGONAL = 2 # maybe add diagonal in the future?

# types of energy functions
cdef uint32_t KL = 1
# cdef uint32_t SYMMETRIC = 2


# define a generic interface for energy and gradient functions
# (e.g. symmetric or KL-divergence).  This defines a type for the function...

# interface for an energy function
# energy(i, j, # word indices to compute energy for
#        mu_ptr,  # pointer to mu array
#        Sigma_ptr,  # pointer to sigma array
#        covariance_type (SPHERICAL, etc)
#        nwords, ndimensions)
ctypedef DTYPE_t (*energy_t)(size_t, size_t,
    DTYPE_t*, DTYPE_t*, uint32_t, size_t, size_t) nogil

# interface for a gradient function
# gradient(i, j, # word indices to compute energy for
#          dEdmui_ptr, dEdSigma1_ptr, dEdmuj_ptr, dEdSigmaj_ptr
#          mu_ptr,  # pointer to mu array
#          Sigma_ptr,  # pointer to sigma array
#          covariance_type (SPHERICAL, etc)
#          nwords, ndimensions)
ctypedef void (*gradient_t)(size_t, size_t,
    DTYPE_t*, DTYPE_t*, DTYPE_t*, DTYPE_t*,
    DTYPE_t*, DTYPE_t*, uint32_t, size_t, size_t) nogil


cdef class GaussianEmbedding:

    cdef np.ndarray mu, Sigma, acc_grad_mu, acc_grad_sigma
    cdef uint32_t covariance_type

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
    cdef DTYPE_t* Sigma_ptr
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
        mu=None, Sigma=None):
        '''
        N = number of distributions (e.g. number of words)
        size = dimension of each Gaussian
        covariance_type = 'spherical' or ...
        energy_type = 'KL' or ...
        mu_max = maximum L2 norm of each mu
        sigma_min, sigma_max = maximum/min eigenvalues of Sigma
        init_params = {
            mu0: initial means are random normal w/ mean=0, std=mu0
            sigma_mean0, sigma_std0: initial Sigma is random normal with
                mean=sigma_mean0, std=sigma_std0
            NOTE: these are ignored if mu/Sigma is explicitly specified
        }
        eta = global learning rate
        Closs = regularization parameter in max-margin loss
            loss = max(0.0, Closs - energy(pos) + energy(neg)
        if mu/Sigma are not None then they specify the initial values
            for the parameters
        '''
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _mu
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] _Sigma
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] _acc_grad_mu
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] _acc_grad_sigma

        if covariance_type == 'spherical':
            self.covariance_type = SPHERICAL
        else:
            raise ValueError

        if energy_type == 'KL':
            self.energy_func = <energy_t>kl_energy
            self.gradient_func = <gradient_t>kl_gradient
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

        if Sigma is None:
            if self.covariance_type == SPHERICAL:
                s = np.random.randn(self.N, 1).astype(DTYPE)
            s *= init_params['sigma_std0']
            s += init_params['sigma_mean0']
            _Sigma = np.ascontiguousarray(
                np.maximum(sigma_min, np.minimum(s, sigma_max)))
        else:
            assert Sigma.shape[0] == N
            if self.covariance_type == SPHERICAL:
                assert Sigma.shape[1] == 1
            _Sigma = Sigma
        self.Sigma = _Sigma

        assert _mu.flags['C_CONTIGUOUS']
        assert _Sigma.flags['C_CONTIGUOUS']
        self.mu_ptr = &_mu[0, 0]
        self.Sigma_ptr = &_Sigma[0, 0]

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

    def nearest_neighbors(self, word_id, metric=cosine, num=10):
        '''Return the num nearest neighbors to word_id, using the metric

        Metric has this interface:
            array(N) with similarities = metric(
                array(N, K) of all vectors, array(K) of word_id
            )
        Returns (top num ids, top num scores)
        '''
        scores = metric(self.mu, self.mu[word_id, :].reshape(1, -1))
        sorted_indices = scores.argsort()[::-1][:(num + 1)]
        top_indices = [ele for ele in sorted_indices if ele != word_id]
        return top_indices, scores[top_indices]

    def __getitem__(self, key):
        if key == 'mu':
            return self.mu
        elif key == 'Sigma':
            return self.Sigma
        elif key == 'acc_grad_mu':
            return self.acc_grad_mu
        elif key == 'acc_grad_sigma':
            return self.acc_grad_sigma
        else:
            raise KeyError
            
    def train_batch(self, np.ndarray[uint32_t, ndim=2, mode='c'] pairs):
        train_batch(&pairs[0, 0], pairs.shape[0],
            self.energy_func, self.gradient_func,
            self.mu_ptr, self.Sigma_ptr, self.covariance_type,
            self.N, self.K,
            self.eta, self.Closs, self.mu_max, self.sigma_min, self.sigma_max,
            self.acc_grad_mu_ptr, self.acc_grad_sigma_ptr
        )

    def energy(self, i, j):
        '''
        Compute the energy between i and j

        This a wrapper around the cython code for convenience
        '''
        return self.energy_func(i, j,
            self.mu_ptr, self.Sigma_ptr, self.covariance_type,
            self.N, self.K)

    def gradient(self, i, j):
        '''
        Compute the gradient with i and j

        This a wrapper around the cython code for convenience
        Returns (dmui, dSigmai), (dmuj, dSigmaj)
        '''
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] dmui
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] dmuj
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] dSigmai
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] dSigmaj

        dmui = np.zeros(self.K, dtype=DTYPE)
        dmuj = np.zeros(self.K, dtype=DTYPE)
        if self.covariance_type == SPHERICAL:
            dSigmai = np.zeros(1, dtype=DTYPE)
            dSigmaj = np.zeros(1, dtype=DTYPE)

        self.gradient_func(i, j,
            &dmui[0], &dSigmai[0], &dmuj[0], &dSigmaj[0],
            self.mu_ptr, self.Sigma_ptr, self.covariance_type,
            self.N, self.K)
        return (dmui, dSigmai), (dmuj, dSigmaj)


cdef DTYPE_t kl_energy(size_t i, size_t j,
    DTYPE_t* mu_ptr, DTYPE_t* Sigma_ptr, uint32_t covariance_type,
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
        det_fac = K * log(Sigma_ptr[j] / Sigma_ptr[i])
        trace_fac = K * Sigma_ptr[j] / Sigma_ptr[i]

        mu_diff_sq = 0.0
        for k in range(K):
            mu_diff_sq += (mu_ptr[i * K + k] - mu_ptr[j * K + k]) ** 2

        return -0.5 * (
            trace_fac
            + mu_diff_sq / Sigma_ptr[i]
            - K - det_fac
        )

cdef void kl_gradient(size_t i, size_t j,
    DTYPE_t* dEdmui_ptr, DTYPE_t* dEdSigmai_ptr,
    DTYPE_t* dEdmuj_ptr, DTYPE_t* dEdSigmaj_ptr,
    DTYPE_t* mu_ptr, DTYPE_t* Sigma_ptr, uint32_t covariance_type,
    size_t N, size_t K) nogil:
    '''
    Implementation of KL-divergence gradient

    dEdmu and dEdSigma are used for return values.
    They must hold enough space for the particular covariance matrix
    (for spherical dEdSigma are floats, for diagonal are K dim arrays
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
            deltaprime = (1.0 / Sigma_ptr[i]) * (
                mu_ptr[i * K + k] - mu_ptr[j * K + k])
            dEdmui_ptr[k] = -deltaprime
            dEdmuj_ptr[k] = deltaprime
            sum_deltaprime2 += deltaprime ** 2

        dEdSigmai_ptr[0] = 0.5 * (
            Sigma_ptr[j] * (1.0 / Sigma_ptr[i]) ** 2
            + sum_deltaprime2
            - (1.0 / Sigma_ptr[i])
        )
        dEdSigmaj_ptr[0] = 0.5 * (1.0 / Sigma_ptr[j] - 1.0 / Sigma_ptr[i])


cdef void train_batch(
    uint32_t* pairs, size_t Npairs,
    energy_t energy_func, gradient_t gradient_func,
    DTYPE_t* mu_ptr, DTYPE_t* Sigma_ptr, uint32_t covariance_type,
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
    cdef DTYPE_t* dSigmai = work + 2 * K
    cdef DTYPE_t* dSigmaj = work + 3 * K

    for k in range(Npairs):

        # compute the loss
        # loss = max(0.0, Closs - energy(pos) + energy(neg))
        posi = pairs[k * 4]
        posj = pairs[k * 4 + 1]
        negi = pairs[k * 4 + 2]
        negj = pairs[k * 4 + 3]

        pos_energy = energy_func(posi, posj,
            mu_ptr, Sigma_ptr, covariance_type, N, K)
        neg_energy = energy_func(negi, negj, 
            mu_ptr, Sigma_ptr, covariance_type, N, K)
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
                dmui, dSigmai, dmuj, dSigmaj,
                mu_ptr, Sigma_ptr, covariance_type, N, K)

            _accumulate_update(i, dmui, dSigmai,
                mu_ptr, Sigma_ptr, covariance_type,
                fac, eta, C, m, M, acc_grad_mu, acc_grad_sigma,
                N, K)
            _accumulate_update(j, dmuj, dSigmaj,
                mu_ptr, Sigma_ptr, covariance_type,
                fac, eta, C, m, M, acc_grad_mu, acc_grad_sigma,
                N, K)

    free(work)

cdef void _accumulate_update(
    size_t k, DTYPE_t* dmu, DTYPE_t* dSigma,
    DTYPE_t* mu_ptr, DTYPE_t* Sigma_ptr, uint32_t covariance_type,
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
        acc_grad_sigma[k] += dSigma[0] ** 2
        local_eta = eta / (sqrt(acc_grad_sigma[k]) + 1.0)
        Sigma_ptr[k] -= fac * local_eta * dSigma[0]
        if Sigma_ptr[k] > M:
            Sigma_ptr[k] = M
        elif Sigma_ptr[k] < m:
            Sigma_ptr[k] = m

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
    # need all positive indices in half_window_size for each word, except
    # words at end, so this slightly overestimates number of pairs
    cdef long long npairs = sum(
        [len(doc) * half_window_size * nsamples_per_word for doc in text]
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
                for k in range(nsamples_per_word):
                    pairs[next_pair, 0] = cdoc[i]
                    pairs[next_pair, 1] = cdoc[j]
                    pairs[next_pair, 2] = cdoc[i]
                    # ignore case where sample is i or j for now
                    pairs[next_pair, 3] = randids[next_pair]
                    next_pair += 1

    return np.ascontiguousarray(pairs[:next_pair, :])


