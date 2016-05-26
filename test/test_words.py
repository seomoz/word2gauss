
import unittest
import numpy as np
import numpy.testing as test
from word2gauss.words import Vocabulary, iter_pairs

DTYPE = np.float32

class TestIterPairs(unittest.TestCase):
    def test_iter_pairs(self):
        np.random.seed(1234)
        vocab = Vocabulary({'zero': 0, 'one': 1, 'two': 2})

        actual = list(iter_pairs(['zero one two', 'one two zero'],
            vocab, batch_size=2, nsamples=1))
        expected = np.array([[0, 1, 0, 2, 0],
                            [0, 1, 1, 1, 1],
                            [0, 2, 0, 0, 0],
                            [0, 2, 0, 2, 1],
                            [1, 2, 1, 0, 0],
                            [1, 2, 1, 2, 1],
                            [1, 2, 1, 1, 0],
                            [1, 2, 1, 2, 1],
                            [1, 0, 1, 2, 0],
                            [1, 0, 2, 0, 1],
                            [2, 0, 2, 2, 0],
                            [2, 0, 0, 0, 1]], dtype=DTYPE)
        self.assertEqual(len(actual), 1)
        self.assertTrue(np.all(actual[0] == expected))


if __name__ == '__main__':
    unittest.main()


