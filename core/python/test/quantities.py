import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, quantities, configuration

import unittest

##########

cfgfile = spirit_py_dir + "/../test/input/fd_neighbours.cfg"   # Input File

p_state = state.setup(cfgfile)                  # State setup

class TestParameters(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
        
class Quantities_Get(TestParameters):
    
    def test_magnetization(self):
        configuration.plus_z(self.p_state)
        M = quantities.get_magnetization(self.p_state)
        self.assertAlmostEqual(M[0], 0)
        self.assertAlmostEqual(M[1], 0)
        self.assertAlmostEqual(M[2], 1)
    
#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Quantities_Get))
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )                         # Delete State

    sys.exit(not success)