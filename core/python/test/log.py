import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import log

import unittest

#########


class TestLog(unittest.TestCase):
    def test_log_message(self):
        with state.State() as p_state:
            log.send(p_state, log.LEVEL_SEVERE, log.SENDER_ALL, "Test Message")


##########


def make_suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(TestLog))
    return suite


if __name__ == "__main__":
    suite = make_suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    sys.exit(not success)
