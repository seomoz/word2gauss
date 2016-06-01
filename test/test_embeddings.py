
import unittest
import numpy as np
import numpy.testing as test
from word2gauss.embeddings import GaussianEmbedding, text_to_pairs
from word2gauss.words import Vocabulary

DTYPE = np.float32

def sample_vocab():
    tokens = {'new':0, 'york':1, 'city':2}
    vocab = Vocabulary(tokens)
    return vocab


def sample_embed(energy_type='KL', covariance_type='spherical', eta=0.1):
    mu = np.array([
        [0.0, 0.0],
        [1.0, -1.25],
        [-0.1, -0.4],
        [1.2, -0.3],
        [0.5, 0.5],
        [-0.55, -0.75]
    ], dtype=DTYPE)
    if covariance_type == 'spherical':
        sigma = np.array([
            [1.0],
            [5.0],
            [0.8],
            [0.4],
            [1.5],
            [1.4]
        ], dtype=DTYPE)
    elif covariance_type == 'diagonal':
        sigma = np.array([
            [1.0, 0.1],
            [5.0, 5.5],
            [0.8, 1.1],
            [0.9, 1.9],
            [0.65, 0.9],
            [1.5, 1.55]
        ], dtype=DTYPE)

    return GaussianEmbedding(3, size=2,
        covariance_type=covariance_type,
        energy_type=energy_type,
        mu=mu, sigma=sigma, eta=eta
    )

class TestSaveLoad(unittest.TestCase):
    def tearDown(self):
        import os
        os.remove(self.tmpname)

    def test_save_load(self):
        import tempfile

        (fid, self.tmpname) = tempfile.mkstemp()

        eta = {'mu':0.1, 'sigma':0.5, 'mu_min': 0.001, 'sigma_min': 0.0005}
        covariance_type = 'diagonal'
        energy_type = 'KL'

        embed = sample_embed(
            covariance_type=covariance_type,
            energy_type=energy_type,
            eta=eta
        )

        embed.save(self.tmpname, full=True)

        # now load and check
        emb = embed.load(self.tmpname)

        self.assertTrue(np.allclose(emb.mu, embed.mu))
        self.assertTrue(np.allclose(emb.sigma, embed.sigma))
        self.assertEqual(emb.covariance_type, embed.covariance_type)
        self.assertEqual(emb.energy_type, embed.energy_type)
        for k, v in emb.eta.items():
            self.assertAlmostEqual(embed.eta[k], v)

class TestKLEnergy(unittest.TestCase):
    def test_kl_energy_spherical(self):
        embed = sample_embed(energy_type='KL', covariance_type='spherical')

        # divergence between same distribution is 0
        self.assertAlmostEqual(embed.energy(1, 1), 0.0)

        # energy = -KL divergence
        # 0 is closer to 2 then to 1
        self.assertTrue(-embed.energy(0, 2) < -embed.energy(0, 1))

    def test_kl_energy_diagonal(self):
        embed = sample_embed(energy_type='KL', covariance_type='diagonal')

        # divergence between same distribution is 0
        self.assertAlmostEqual(embed.energy(1, 1), 0.0)

        # energy = -KL divergence
        # 0 is closer to 2 then to 1
        self.assertTrue(-embed.energy(0, 2) < -embed.energy(0, 1))


class TestIPEnergy(unittest.TestCase):
    # energy is log(P(0; mui - muj, Sigmai + Sigmaj)
    # use scipy's multivariate_normal to get true probability
    # then take log

    def test_ip_energy_spherical(self):
        from scipy.stats import multivariate_normal

        embed = sample_embed(energy_type='IP', covariance_type='spherical')

        mui = embed.mu[1, :]
        muj = embed.mu[2, :]
        sigma = np.diag(
            (embed.sigma[1] + embed.sigma[2]) * np.ones(2))
        expected = np.log(multivariate_normal.pdf(
            np.zeros(2), mean=mui - muj, cov=sigma))
        actual = embed.energy(1, 2)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_ip_energy_diagonal(self):
        from scipy.stats import multivariate_normal

        embed = sample_embed(energy_type='IP', covariance_type='diagonal')

        mui = embed.mu[1, :]
        muj = embed.mu[2, :]
        sigma = np.diag(embed.sigma[1, :] + embed.sigma[2, :])
        expected = np.log(multivariate_normal.pdf(
            np.zeros(2), mean=mui - muj, cov=sigma))
        actual = embed.energy(1, 2)
        self.assertAlmostEqual(actual, expected, places=6)

def numerical_grad(embed, i, j, eps=1.0e-3):
    '''
    Computes gradient and numerical gradient

    returns [(grad mu, numerical grad mu), (grad sigma), (num. grad sigma)]
    '''
    from word2gauss.embeddings import COV_MAP

    # compute the gradient at i, j
    (dmui, dsigmai), (dmuj, dsigmaj) = embed.gradient(i, j)
    dmu = [dmui, dmuj]
    dsigma = [dsigmai, dsigmaj]

    # now compute numerical gradient
    Eij = embed.energy(i, j)

    ndmu = [np.zeros(dmui.shape), np.zeros(dmuj.shape)]
    ndsigma = [np.zeros(dsigmai.shape), np.zeros(dsigmaj.shape)]
    
    for ind, ij in enumerate([i, j]):
        for k in xrange(embed.K):
            embed.mu[ij, k] += eps
            E = embed.energy(i, j)
            ndmu[ind][k] = (E - Eij) / eps
            embed.mu[ij, k] -= eps

            if COV_MAP[embed.covariance_type] == 'diagonal':
                embed.sigma[ij, k] += eps
                E = embed.energy(i, j)
                ndsigma[ind][k] = (E - Eij) / eps
                embed.sigma[ij, k] -= eps

        if COV_MAP[embed.covariance_type] == 'spherical':
            embed.sigma[ij] += eps
            E = embed.energy(i, j)
            ndsigma[ind] = (E - Eij) / eps
            embed.sigma[ij] -= eps


    return [(dmu, ndmu), (dsigma, ndsigma)]


class TestNumericalGradient(unittest.TestCase):
    def _num_grad_check(self, embed, eps, rtol):
        [(dmu, ndmu), (dsigma, ndsigma)] = numerical_grad(embed, 0, 1, eps)
        for ij in [0, 1]:
            self.assertTrue(
                np.allclose(dmu[ij], ndmu[ij], rtol=rtol))
            self.assertTrue(
                np.allclose(dsigma[ij], ndsigma[ij], rtol=rtol))

    def test_numerical_grad_kl(self):
        embed = sample_embed('KL', 'spherical')
        self._num_grad_check(embed, 1.0e-3, 1e-1)

        embed = sample_embed('KL', 'diagonal')
        self._num_grad_check(embed, 1.0e-3, 1e-1)

    def test_numerical_grad_ip(self):
        embed = sample_embed('IP', 'spherical')
        self._num_grad_check(embed, 1.0e-3, 1e-1)

        embed = sample_embed('IP', 'diagonal')
        self._num_grad_check(embed, 1.0e-3, 1e-1)


class TestGaussianEmbedding(unittest.TestCase):
    def _training_data(self):
        # 10 words
        # word 0 and 1 co-occur frequently
        # the rest co-occur randomly

        np.random.seed(5)

        # number of sample to do
        nsamples = 100000
        training_data = np.empty((nsamples, 5), dtype=np.uint32)
        for k in xrange(nsamples):
            i = np.random.randint(0, 10)

            # the positive sample
            if i == 0 or i == 1:
                # choose the other 50% of the time
                if np.random.rand() < 0.5:
                    j = 1 - i
                else:
                    j = np.random.randint(0, 10)
            else:
                j = np.random.randint(0, 10)
            pos = (i, j)

            # the negative sample
            neg = (i, np.random.randint(0, 10))

            # randomly sample whether left or right is context word
            context_index = np.random.randint(0, 2)

            training_data[k, :] = pos + neg + (context_index, )

        return training_data

    def _check_results(self, embed):
        # should have 0 - 1 close together and 0..1 - 2..9 far apart
        # should also have 2..9 all near each other
        neighbors0 = embed.nearest_neighbors(0, num=10)
        # neighbors[0] is 0
        self.assertEqual(neighbors0[1]['id'], 1)

        # check nearest neighbors to 2, the last two should be 0, 1
        neighbors2 = embed.nearest_neighbors(2, num=10)
        last_two_ids = sorted([result['id'] for result in neighbors2[-2:]])
        self.assertEqual(sorted(last_two_ids), [0, 1])

    def test_model_update(self):
        for covariance_type, sigma_shape1 in [
                ('spherical', 1), ('diagonal', 2)]:
            embed = sample_embed(covariance_type=covariance_type)
            embed.update(5)

            self.assertEquals(embed.mu.shape, (10, 2))
            self.assertEquals(embed.sigma.shape, (10, sigma_shape1))
            self.assertEquals(embed.acc_grad_mu.shape, (10, 2))
            self.assertEquals(embed.acc_grad_sigma.shape, (10, sigma_shape1))

            self.assertEquals(embed.N, 5)

    def test_train_batch_KL_spherical(self):
        training_data = self._training_data()

        embed = GaussianEmbedding(10, 5,
            covariance_type='spherical',
            energy_type='KL',
            mu_max=2.0, sigma_min=0.8, sigma_max=1.0, eta=0.1, Closs=1.0
        )

        for k in xrange(0, len(training_data), 100):
            embed.train_batch(training_data[k:(k+100)])

        self._check_results(embed)

    def test_train_batch_KL_diagonal(self):
        training_data = self._training_data()
        embed = GaussianEmbedding(10, 5,
            covariance_type='diagonal',
            energy_type='KL',
            mu_max=2.0, sigma_min=0.8, sigma_max=1.2, eta=0.1, Closs=1.0
        )

        # diagonal training has more parameters so needs more then one
        # epoch to fully learn data
        for k in xrange(0, len(training_data), 100):
            embed.train_batch(training_data[k:(k+100)])

        self._check_results(embed)


    def test_phrases_to_vector1(self):
        self.embed = sample_embed(energy_type='IP',
            covariance_type='spherical')
        vocab = sample_vocab()
        target = [["new"], ["york"]]
        res = np.array([-1. , 1.25])
        vec = self.embed.phrases_to_vector(target, vocab=vocab)
        test.assert_array_equal(vec, res)

    def test_phrases_to_vector2(self):
        self.embed = sample_embed(energy_type='IP',
            covariance_type='spherical')
        vocab = sample_vocab()
        target = [["new"], []]
        res = np.array([0. , 0])
        vec = self.embed.phrases_to_vector(target, vocab=vocab)
        test.assert_array_equal(vec, res)

    def test_phrases_to_vector3(self):
        self.embed = sample_embed(energy_type='IP', covariance_type='spherical')
        vocab = sample_vocab()
        target = [["new"], [""]]
        res = np.array([0. , 0])
        vec = self.embed.phrases_to_vector(target, vocab=vocab)
        test.assert_array_equal(vec, res)

    def test_train_batch_IP_spherical(self):
        training_data = self._training_data()

        embed = GaussianEmbedding(10, 5,
            covariance_type='spherical',
            energy_type='IP',
            mu_max=2.0, sigma_min=0.8, sigma_max=1.2, eta=0.1, Closs=1.0
        )

        for k in xrange(0, len(training_data), 100):
            embed.train_batch(training_data[k:(k+100)])

        self._check_results(embed)

    def test_train_batch_IP_diagonal(self):
        training_data = self._training_data()

        embed = GaussianEmbedding(10, 5,
            covariance_type='diagonal',
            energy_type='IP',
            mu_max=2.0, sigma_min=0.8, sigma_max=1.2, eta=0.1, Closs=1.0
        )

        for k in xrange(0, len(training_data), 100):
            embed.train_batch(training_data[k:(k+100)])

        self._check_results(embed)

    def test_train_threads(self):
        training_data = self._training_data()

        embed = GaussianEmbedding(10, 5,
            covariance_type='spherical',
            energy_type='KL',
            mu_max=2.0, sigma_min=0.8, sigma_max=1.2, eta=0.1, Closs=1.0
        )

        def iter_pairs():
            for k in xrange(0, len(training_data), 100):
                yield training_data[k:(k+100)]

        embed.train(iter_pairs(), n_workers=4)

        self._check_results(embed)

    def test_eta_single(self):
        embed = GaussianEmbedding(10, 5, eta=0.55)
        expected = {
            'mu': 0.55,
            'mu_min': 0.0,
            'sigma': 0.55,
            'sigma_min': 0.0
        }
        actual = embed.eta
        for k, v in expected.items():
            self.assertAlmostEqual(actual[k], v)

    def test_eta_multiple(self):
        expected = {
            'mu': 0.1,
            'mu_min': 0.001,
            'sigma': 0.05,
            'sigma_min': 0.000005
        }
        embed = GaussianEmbedding(10, 5, eta=expected)
        actual = embed.eta
        for k, v in expected.items():
            self.assertAlmostEqual(actual[k], v)


class TestTexttoPairs(unittest.TestCase):
    def test_text_to_pairs(self):
        # mock out the random int generator
        r = lambda N: np.arange(N, dtype=np.uint32)
        text = [
            np.array([1, 2, 3, -1, -1, 4, 5], dtype=np.uint32),
            np.array([], dtype=np.uint32),
            np.array([10, 11], dtype=np.uint32)
        ]
        actual = text_to_pairs(text, r, nsamples_per_word=2)
        expected = np.array([[ 1,  2,  1,  0, 0],
           [ 1,  2,  1,  2, 1],
           [ 1,  2,  1,  2, 0],
           [ 1,  2,  3,  2, 1],
           [ 1,  3,  1,  4, 0],
           [ 1,  3,  5,  3, 1],
           [ 1,  3,  1,  6, 0],
           [ 1,  3,  7,  3, 1],
           [ 2,  3,  2,  8, 0],
           [ 2,  3,  9,  3, 1],
           [ 2,  3,  2, 10, 0],
           [ 2,  3, 11,  3, 1],
           [ 4,  5,  4, 12, 0],
           [ 4,  5, 13,  5, 1],
           [ 4,  5,  4, 14, 0],
           [ 4,  5, 15,  5, 1],
           [10, 11, 10, 16, 0],
           [10, 11, 17, 11, 1],
           [10, 11, 10, 18, 0],
           [10, 11, 19, 11, 1]], dtype=np.uint32)
        self.assertTrue((actual == expected).all())


if __name__ == '__main__':
    unittest.main()


