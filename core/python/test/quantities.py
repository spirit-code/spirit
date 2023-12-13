import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, quantities, configuration, geometry, version

import unittest

##########

cfgfile = spirit_py_dir + "/../test/input/api.cfg"  # Input File

p_state = state.setup(cfgfile)  # State setup


class TestParameters(unittest.TestCase):
    def setUp(self):
        """Setup a p_state and copy it to Clipboard"""
        self.p_state = p_state

        self.precision_apprx = 7
        self.precision_rough = 7
        if version.scalartype == "float":
            print(
                "\nWARNING: Detected single precision calculation. Reducing precision requirements.\n"
            )
            self.precision_apprx = 6
            self.precision_rough = 4


class Quantities_Get(TestParameters):
    def test_magnetization(self):
        configuration.plus_z(self.p_state)
        mu_s = 1.34
        geometry.set_mu_s(self.p_state, mu_s)
        M = quantities.get_magnetization(self.p_state)
        self.assertAlmostEqual(M[0], 0)
        self.assertAlmostEqual(M[1], 0)
        self.assertAlmostEqual(M[2], mu_s, self.precision_rough)

    def test_topological_charge(self):
        configuration.plus_z(self.p_state)
        configuration.skyrmion(self.p_state, radius=5, pos=[1.5, 0, 0])
        Q = quantities.get_topological_charge(self.p_state)
        [Q_density, triangles] = quantities.get_topological_charge_density(self.p_state)
        self.assertAlmostEqual(Q, -1.0, 6)
        self.assertAlmostEqual(Q, sum(Q_density), self.precision_apprx)


#########


def make_suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(Quantities_Get))
    return suite


if __name__ == "__main__":
    suite = make_suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete(p_state)  # Delete State

    sys.exit(not success)
