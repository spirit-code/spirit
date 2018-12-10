import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import configuration

import unittest

##########

cfgfile = spirit_py_dir + "/../test/input/api.cfg"

class TestConfigurations(unittest.TestCase):
    def test(self):
        with state.State(cfgfile) as p_state:
            # Noise
            configuration.random(p_state)
            configuration.add_noise(p_state, 5)
            # Homogeneous
            configuration.plus_z(p_state)
            configuration.minus_z(p_state)
            configuration.domain(p_state, [1,1,1])
            # Skyrmion
            configuration.skyrmion(p_state, 5)
            # Hopfion
            configuration.hopfion(p_state, 5)
            # Spin Spiral
            configuration.spin_spiral(p_state, "Real Lattice", [0,0,0.1], [0,0,1], 30)

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestConfigurations))
  
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    sys.exit(not success)