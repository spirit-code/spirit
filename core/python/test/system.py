import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, system, configuration

import unittest

##########

cfgfile = spirit_py_dir + "/../test/input/fd_neighbours.cfg"   # Input File
p_state = state.setup(cfgfile)                  # State setup

class TestSystem(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
        
class SystemGetters(TestSystem):
    
    def test_get_index(self):
        index = system.get_index(self.p_state)
        self.assertEqual(index, 0)
    
    def test_get_nos(self):
        nos = system.get_nos(self.p_state)
        self.assertEqual(nos, 4)
    
    def test_get_spin_directions(self):
        configuration.plus_z(self.p_state)
        nos = system.get_nos(self.p_state)
        arr = system.get_spin_directions(self.p_state)
        for i in range(nos):
            self.assertAlmostEqual( arr[i][0], 0. )
            self.assertAlmostEqual( arr[i][1], 0. )
            self.assertAlmostEqual( arr[i][2], 1. )
    
    def test_get_energy(self):
        # NOTE: that test is trivial
        E = system.get_energy(self.p_state)
    
    
    # NOTE: there is no way to test the system.Update_Data() and system.Print_Energy_Array()

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SystemGetters))
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )                         # delete state

    sys.exit(not success)