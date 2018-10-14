import numpy as np
import pandas as pd
import unittest
import math
from logger import logger
from metrics import *

class Metrics(unittest.TestCase):
    def setUp(self):
        ls = [1, 1, 1, 1, 2, 2, 3]
        self.dataset = [
            ls,
            pd.DataFrame(ls),
            np.array(ls)
        ]
    def test_Gini(self):
        ls = [(4/7) ** 2, (2/7) ** 2, (1/7) ** 2]
        for d in self.dataset:
            self.assertEqual(Gini(d), 1-sum(ls))
    def test_Entropy(self):
        ls = [(4/7), (2/7), (1/7)]
        ls2 = [i * math.log2(i) for i in ls ]
        for d in self.dataset:
            self.assertEqual(Entropy(d), - sum(ls2))

class LogisticFunction(unittest.TestCase):
    def setUp(self):
        self.x1 = np.array([1, -1, 2, -2, 3, -3, 4, -4])
        self.x2 = np.array([
            [1, 2, -1],
            [-2, 3, -1],
            [1, -2, 1]
        ])
    def test_ReLu(self):
        x1_relu = ReLu(self.x1)
        res = np.all(x1_relu == np.array([1, 0, 2, 0, 3, 0, 4, 0]))
        self.assertTrue(res)

        x2_relu = ReLu(self.x2)
        exp = np.array([
            [1, 2, 0],
            [0, 3, 0],
            [1, 0, 1]
        ])
        res = np.all(np.all(exp == x2_relu))
        self.assertTrue(res)
    def test_leak_relu(self):
        x1_lrelu = LeakReLu(0.1, self.x1)
        exp = np.array([1, -0.1, 2, -0.2, 3, -0.3, 4, -0.4])
        res = np.all(exp - x1_lrelu < 1e-8)
        self.assertTrue(res)

        x2_lrelu = LeakReLu(0.2, self.x2)
        exp = np.array([
            [1, 2, -0.2],
            [-0.4, 3, -0.2],
            [1, -0.4, 1]
        ])
        res = np.all(exp == x2_lrelu)
        self.assertTrue(res)
if __name__ == "__main__":
    unittest.main()

