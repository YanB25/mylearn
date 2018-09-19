from logger.logger import logger
import unittest

class MyTest(unittest.TestCase):
    def setUp(self):
        logger.debug('begin to test')
    def test_int(self):
        self.assertEqual(1, 1)

if __name__ == "__main__":
    unittest.main()

