import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import hamiltonian

import unittest

##########

cfgfile = spirit_py_dir + "/../test/input/fd_neighbours.cfg"   # Input File

p_state = state.setup(cfgfile)                  # State setup

class TestParameters(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
        
class Hamiltonian_set_get(TestParameters):
    
    def test_set_field(self):
        mag_set = 10
        dir_set = [1., 1., 0.]
        hamiltonian.set_field(self.p_state, mag_set, dir_set)
        # TODO: test the functionality of that function when the corresponding get will be available

    def test_get_field(self):
        import math
        self.test_set_field() # set to ensure the right values are actually set
        magnitude, normal = hamiltonian.get_field(self.p_state)
        self.assertAlmostEqual( magnitude, 10 )
        expected_normal = [1/math.sqrt(2), 1/math.sqrt(2), 0] # 1/sqrt(2) because of normalization
        self.assertAlmostEqual( normal[0], expected_normal[0])
        self.assertAlmostEqual( normal[1], expected_normal[1])
        self.assertAlmostEqual( normal[2], expected_normal[2])

    def test_set_anisotropy(self):
        mag_set = 0.5
        dir_set = [1., 1., 0.]
        hamiltonian.set_anisotropy(self.p_state, mag_set, dir_set)
        # TODO: test the functionality of that function when the corresponding get will be available
        
    
#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Hamiltonian_set_get))
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )                         # Delete State

    sys.exit(not success)