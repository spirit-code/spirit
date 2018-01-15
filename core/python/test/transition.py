import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, chain, transition

import unittest

##########

cfgfile = spirit_py_dir + "/../../input/input.cfg"                 # Input File
p_state = state.setup(cfgfile)              # State setup
chain.Image_to_Clipboard(p_state)         # Copy p_state to Clipboard

class TestTransition(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
        chain.Insert_Image_After(self.p_state)      # image 1st
        chain.Insert_Image_After(self.p_state)      # image 2nd
        chain.Insert_Image_After(self.p_state)      # image 3rd
    
    def tearDown(self):
        ''' clean the p_state '''
        noi = chain.Get_NOI(self.p_state)
        for i in range(noi-1):
            chain.Pop_Back(self.p_state)
        self.assertEqual(chain.Get_NOI(self.p_state), 1)
    
class trivialTestTransition(TestTransition):
    
    def test_homogeneous(self):
        transition.Homogeneous(self.p_state, 1, 2)
    
    def test_add_noise_temperature(self):
        temperature = 100
        transition.Add_Noise_Temperature(self.p_state, 100, 1, 2)
    
##########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(trivialTestTransition))
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )                         # Delete State

    sys.exit(not success)