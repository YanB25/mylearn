import numpy as np
import unittest
from logger import logger
from metrics import *

class Metrics(unittest.TestCase):
    def setUp(self):
        self.dataset = [1, 1, 1, 1, 2, 2, 3]
    def test_Gini(self):
        ls = [(4/7) ** 2, (2/7) ** 2, (1/7) ** 2]
        self.assertEqual(Gini(self.dataset), 1-sum(ls))


if __name__ == "__main__":
    unittest.main()

