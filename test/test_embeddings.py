
import unittest

import numpy as np

from word2gauss.embeddings import GaussianEmbedding
from word2gauss.utils import cosine

DTYPE = np.float32

class TestKLEnergy(unittest.TestCase):
    def setUp(self):
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

        self.embed = GaussianEmbedding(3, size=2,
            covariance_type='spherical',
            energy_type='KL',
            mu=mu, Sigma=Sigma
        )

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


if __name__ == '__main__':
    unittest.main()


