import unittest

import numpy as np

from run_batched import run_batched


def instrumented_double(x, batch_size_callback):
    batch_size_callback(len(x))
    return 2 * x


class TestBasic(unittest.TestCase):
    def test_addition(self):
        batch_sizes = []
        self.assertEqual(
            run_batched(
                lambda x: instrumented_double(x, batch_sizes.append),
                np.array([1, 2, 3]),
                2,
                device="cpu",
            ).tolist(),
            [2, 4, 6],
        )
        self.assertEqual(batch_sizes, [2, 1])
