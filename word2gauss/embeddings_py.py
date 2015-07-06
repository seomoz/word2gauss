'''
Numpy/Python implementation of word2gauss, used as initial prototype
for the Cython version.  Eventually this will be removed
'''


import numpy as np

DTYPE = np.float32

class GaussianDistribution(object):
    def __init__(self, N, size=100, covariance_type='spherical'):
        '''
        Holds N size-dimensional Guassians with specified covariance structure

        N = number of distributions (e.g. number of words)
        size = dimension of each Gaussian
        '''
        self.covariance_type = covariance_type

        self.N = N 
        self.K = size

        # allocate space for parameters
        self.mu = np.empty((self.N, self.K), dtype=DTYPE)
        if self.covariance_type == 'spherical':
            self.Sigma = np.empty((self.N, 1), dtype=DTYPE)
        elif self.covariance_type == 'diagonal':
            self.Sigma = np.empty((self.N, self.K), dtype=DTYPE)

    def init_params(self, mu0, sigma_mean0, sigma_std0, sigma_min, sigma_max):
        '''
        Initialize parameters 

        mu = random normal with std mu0, mean 0
        Sigma = random normal with mean sigma_mean0, std sigma_std0,
            and min / max of sigma_min, sigma_max
        '''
        self.mu = mu0 * np.random.randn(self.N, self.K).astype(DTYPE)
        self.Sigma = np.random.randn(*self.Sigma.shape).astype(DTYPE)
        self.Sigma *= sigma_std0
        self.Sigma += sigma_mean0
        self.Sigma = np.maximum(sigma_min, np.minimum(self.Sigma, sigma_max))


class KLEnergy(object):
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

    def __init__(self, dist):
        self._dist = dist
        # even though we have SOME code for diagonal case,
        # spherical is only case tested for now
        assert self._dist.covariance_type == 'spherical'

    def _params_from_indices(self, i, j):
        mu_i = self._dist.mu[i]
        mu_j = self._dist.mu[j]
        Sigma_i = self._dist.Sigma[i]
        Sigma_j = self._dist.Sigma[j]
        return mu_i, mu_j, Sigma_i, Sigma_j

    def energy(self, i, j):
        '''
        E(P[i], P[j]) = Negative KL divergence
        '''
        mu_i, mu_j, Sigma_i, Sigma_j = self._params_from_indices(i, j)

        if self._dist.covariance_type == 'spherical':
            # log(det(sigma[j] / sigma[i]) = log det sigmaj - log det sigmai
            #   det = sigma ** K  SO
            # = K * (log(sigmaj) - log(sigmai)) = K * log(sigma j / sigmai)
            det_fac = self._dist.K * np.log(Sigma_j / Sigma_i)
            trace_fac = self._dist.K * Sigma_j / Sigma_i
        elif self._dist.covariance_type == 'diagonal':
            # log prod(sigmaj) - log prod(sigmai) =
            # sum log(sigmaj) - sum(log(sigmai))
            det_fac = np.sum(np.log(Sigma_i)) - np.sum(np.log(Sigma_j))
            trace_fac = np.sum(Sigma_j / Sigma_i)

        return -0.5 * float(  # float to get out a single number, not 1x1 array
            trace_fac
            + np.sum((mu_i - mu_j) ** 2 / Sigma_i)
            - self._dist.K - det_fac
        )

    def gradient(self, i, j):
        # returns:
        # (dE/dmui, dE/dSigmai), (dE/dmuj, dE/dSigmaj)
        # dE/dmui and dE/dmuj = (K, 1) vector
        # dE/dSigmaj and dE/dSigmaj and same shape as Sigmai (float or K, 1))
        mu_i, mu_j, Sigma_i, Sigma_j = self._params_from_indices(i, j)
        if self._dist.covariance_type == 'spherical':
            deltaprime = (1.0 / Sigma_i) * (mu_i - mu_j)
            dEdi = 0.5 * (
                Sigma_j * (1.0 / Sigma_i) ** 2
                + np.sum(deltaprime ** 2)
                - (1.0 / Sigma_i)
            )
            dEdj = 0.5 * (1.0 / Sigma_j - 1.0 / Sigma_i)
        return (-deltaprime, dEdi), (deltaprime, dEdj)


class GaussianEmbedding(object):
    def __init__(self, N, size=100, covariance_type='spherical',
        energy='KL', C=1.0, m=0.1, M=10.0, Closs=1.0, eta=1.0):

        '''
        N = number of tokens
        size = dimensionality of each embedding
        covariance_type = 'spherical' only now (maybe diag in future)
        energy = 'KL' (maybe also add symmetric inner product)

        Closs = regularization constant in max-margin loss
        C = regularization constant on l2-norm of means
        m, M  = min / max eigenvalues of co-variance matrix
        eta = global learning rate
        '''

        self.dist = GaussianDistribution(N,
            size=size,
            covariance_type=covariance_type
        )
        if energy == 'KL':
            self.energy = KLEnergy(self.dist)
        else:
            raise NotImplementedError

        self.C = C
        self.m = m
        self.M = M
        self.Closs = Closs

        # initialize parameters
        self.dist.init_params(0.1, M, 1.0, m, M)

        # learning rates
        self.eta = eta
        # for adagrad, need accumulated gradient for each mu, sigma
        self._acc_grad_mu = np.zeros(N)
        self._acc_grad_sigma = np.zeros(N)

    def _loss(self, pos, neg):
        # max(0, C - E(P[i_pos], P[j_pos]) + E(P[i_neg, P[i_neg]))
        return max(0.0,
            self.Closs - self.energy.energy(*pos) + self.energy.energy(*neg)
        )

    def train_single(self, pairs):
        '''
        pairs = an iterator of (i_pos, j_pos), (i_neg, j_neg) pairs
        to use for training

        updates parameters on pair at a time
        '''
        for pos, neg in pairs:
            loss = self._loss(pos, neg)
            if loss < 1e-14:
                # loss for this sampe is 0, nothing to update
                continue

            # compute gradients and update
            for pn, fac in [(pos, -1.0), (neg, 1.0)]:
                i, j = pn
                di, dj = self.energy.gradient(i, j)
                # di = (dE/dmui, dE/dSigmai) and same for dj
                for k, d in [(i, di), (j, dj)]:

                    # accumulate gradients and update
                    dmu, dsigma = d
                    self._acc_grad_mu[k] += np.sum(dmu ** 2) / len(dmu)
                    eta = self.eta / np.sqrt(self._acc_grad_mu[k] + 1.0)
                    self.dist.mu[k] -= fac * eta * dmu
                    # regularizer
                    l2_mu = np.sqrt(np.sum(self.dist.mu[k] ** 2))
                    if l2_mu > self.C:
                        self.dist.mu[k] *= (self.C / l2_mu)

                    self._acc_grad_sigma[k] += np.sum(dsigma ** 2) / len(dsigma)
                    eta = self.eta / np.sqrt(self._acc_grad_sigma[k] + 1.0)
                    self.dist.Sigma[k] -= fac * eta * dsigma
                    # regularizer
                    self.dist.Sigma[k] = np.maximum(self.m,
                        np.minimum(self.M, self.dist.Sigma[k]))

