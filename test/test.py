import numpy as np
import unittest
from logger.logger import logger
from tree import DecisionTreeClassifier

class MyTest(unittest.TestCase):
    def setUp(self):
        logger.debug('begin to test')
    def test_int(self):
        self.assertEqual(1, 1)

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.X = [1, 2, 3]
        self.Y = [2, 4, 5]
        self.predict_Y = [1, 4, 6]
    def test_buildtree(self):
        dtc = DecisionTreeClassifier()
        dtc.fit(self.X, self.Y)
        dtc.predict(self.predict_Y)

if __name__ == "__main__":
    unittest.main()

