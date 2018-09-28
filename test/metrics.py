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

if __name__ == "__main__":
    unittest.main()

