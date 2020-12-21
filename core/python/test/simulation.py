import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, simulation, configuration, version
from spirit.parameters import llg
import unittest

##########

LLG = simulation.METHOD_LLG
SIB = simulation.SOLVER_SIB

##########

cfgfile = spirit_py_dir + "/../test/input/solvers.cfg"     # Input File

p_state = state.setup(cfgfile)              # State setup

class TestParameters(unittest.TestCase):
        
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
        if version.scalartype == "float":
            print("\nWARNING: Detected single precision calculation. Reducing precision requirements.\n")
            llg.set_convergence(p_state, 1e-5)


class Simulation_StartStop(TestParameters):
    
    def test_singleshot(self):
        configuration.plus_z(self.p_state)
        simulation.start(self.p_state, LLG, SIB, n_iterations=1, single_shot=True)
        simulation.single_shot(self.p_state)
        simulation.start(self.p_state, LLG, SIB, single_shot=True)
        simulation.single_shot(self.p_state)
        simulation.stop(self.p_state)

    def test_playpause(self):
        configuration.plus_z(self.p_state)
        configuration.skyrmion(p_state, 5)
        simulation.start(self.p_state, LLG, SIB)

    def test_stopall(self):
        simulation.stop_all(self.p_state)

class Simulation_Running(TestParameters):
    
    def test_running_image(self):
        self.assertFalse(simulation.running_on_image(self.p_state))

    def test_running_chain(self):
        self.assertFalse(simulation.running_on_chain(self.p_state))

    def test_running_anywhere_chain(self):
        self.assertFalse(simulation.running_anywhere_on_chain(self.p_state))

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Simulation_StartStop))
    suite.addTest(unittest.makeSuite(Simulation_Running))
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )                         # Delete State

    sys.exit(not success)