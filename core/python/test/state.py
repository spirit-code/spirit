import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, configuration

import unittest

##########

cfgfile = spirit_py_dir + "/../../input/input.cfg"


class TestState(unittest.TestCase):
    def test(self):
        with state.State(cfgfile) as p_state:
            pass


#########


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestState))

    return suite


if __name__ == "__main__":
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()
    sys.exit(not success)
