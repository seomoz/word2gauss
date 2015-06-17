
import unittest

import numpy as np

from word2gauss.embeddings import GaussianEmbedding, text_to_pairs

DTYPE = np.float32

def sample_embed():
    mu = np.array([
        [0.0, 0.0],
        [1.0, -1.25],
        [-0.1, -0.4]
    ], dtype=DTYPE)
    Sigma = np.array([
        [1.0],
        [5.0],
        [0.8]
    ], dtype=DTYPE)

    return GaussianEmbedding(3, size=2,
        covariance_type='spherical',
        energy_type='KL',
        mu=mu, Sigma=Sigma
    )

class TestKLEnergy(unittest.TestCase):
    def setUp(self):
        self.embed = sample_embed()

    def test_kl_energy_spherical(self):
        # divergence between same distribution is 0
        self.assertAlmostEqual(self.embed.energy(1, 1), 0.0)

        # energy = -KL divergence
        # 0 is closer to 2 then to 1
        self.assertTrue(-self.embed.energy(0, 2) < -self.embed.energy(0, 1))

class TestGaussianEmbedding(unittest.TestCase):
    def test_train(self):
        # 10 words
        # word 0 and 1 co-occur frequently
        # the rest co-occur randomly

        np.random.seed(5)

        # number of sample to do
        nsamples = 100000
        training_data = np.empty((nsamples, 4), dtype=np.uint32)
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

            training_data[k, :] = pos + neg

        embed = GaussianEmbedding(10, 5,
            covariance_type='spherical',
            energy_type='KL',
            mu_max=1.0, sigma_min=0.1, sigma_max=1.0, eta=1.0, Closs=1.0
        )

        for k in xrange(0, nsamples, 100):
            embed.train_batch(training_data[k:(k+100)])

        # should have 0 - 1 close together and 0..1 - 2..9 far apart
        # should also have 2..9 all near each other
        neighbors0, scores0 = embed.nearest_neighbors(0, num=10)
        self.assertEqual(neighbors0[0], 1)

        # check nearest neighbors to 2, the last two should be 0, 1
        neighbors2, scores2 = embed.nearest_neighbors(2, num=10)
        self.assertEqual(sorted(neighbors2[-2:]), [0, 1])


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
        expected = np.array([[ 1,  2,  1,  0],
               [ 1,  2,  1,  1],
               [ 1,  3,  1,  2],
               [ 1,  3,  1,  3],
               [ 2,  3,  2,  4],
               [ 2,  3,  2,  5],
               [ 4,  5,  4,  6],
               [ 4,  5,  4,  7],
               [10, 11, 10,  8],
               [10, 11, 10,  9]], dtype=np.uint32)
        self.assertTrue((actual == expected).all())


if __name__ == '__main__':
    unittest.main()


