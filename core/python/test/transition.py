import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, chain, transition

import unittest

##########

p_state = state.setup()           # State setup
chain.image_to_clipboard(p_state) # Copy p_state to Clipboard

class TestTransition(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
        chain.insert_image_after(self.p_state)      # image 1st
        chain.insert_image_after(self.p_state)      # image 2nd
        chain.insert_image_after(self.p_state)      # image 3rd
    
    def tearDown(self):
        ''' clean the p_state '''
        noi = chain.get_noi(self.p_state)
        for i in range(noi-1):
            chain.pop_back(self.p_state)
        self.assertEqual(chain.get_noi(self.p_state), 1)
    
class trivialTestTransition(TestTransition):
    
    def test_homogeneous(self):
        transition.homogeneous(self.p_state, 1, 2)
    
    def test_add_noise(self):
        temperature = 100
        transition.add_noise(self.p_state, 100, 1, 2)
    
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