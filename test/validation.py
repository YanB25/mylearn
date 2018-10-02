import numpy as np
import pandas as pd
import unittest
from tree import DecisionTreeClassifier
from logger import logger
from model_selection import *

class Validation(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [5, 6, 7, 8, 9],
            [5, 3, 1, 6, 7],
            [1, 2, 5, 6, 3]
        ])
        self.Y = np.array([1, 0, 1, 0, 0])
    def test_run(self):
        clf = DecisionTreeClassifier()
        cross_val_score(clf, self.X, self.Y, cv=2)




if __name__ == "__main__":
    unittest.main()

