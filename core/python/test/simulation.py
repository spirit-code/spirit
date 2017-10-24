import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, simulation, configuration

import unittest

##########

cfgfile = "core/test/input/solvers.cfg"     # Input File

p_state = state.setup(cfgfile)              # State setup

class TestParameters(unittest.TestCase):
        
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
    
class Simulation_StartStop(TestParameters):
    
    def test_singleshot(self):
        configuration.PlusZ(self.p_state)
        simulation.SingleShot(self.p_state, "LLG", "SIB", n_iterations=1)
    
    def test_playpause(self):
        configuration.PlusZ(self.p_state)
        configuration.Skyrmion(p_state, 5)
        simulation.PlayPause(self.p_state, "LLG", "SIB")
    
    def test_stopall(self):
        simulation.Stop_All(self.p_state)

class Simulation_Running(TestParameters):
    
    def test_running_image(self):
        self.assertFalse(simulation.Running_Image(self.p_state))
    
    def test_running_chain(self):
        self.assertFalse(simulation.Running_Chain(self.p_state))
    
    def test_running_collection(self):
        self.assertFalse(simulation.Running_Collection(self.p_state))
    
    def test_running_anywhere_chain(self):
        self.assertFalse(simulation.Running_Anywhere_Chain(self.p_state))
    
    def test_running_anywhere_collection(self):
        self.assertFalse(simulation.Running_Anywhere_Collection(self.p_state))
    
#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Simulation_StartStop))
    suite.addTest(unittest.makeSuite(Simulation_Running))
    return suite

suite = suite()

runner = unittest.TextTestRunner()
success = runner.run(suite).wasSuccessful()

state.delete( p_state )                         # Delete State

sys.exit(not success)