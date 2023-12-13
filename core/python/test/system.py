import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, system, configuration, hamiltonian

import unittest

##########

cfgfile = spirit_py_dir + "/../test/input/fd_neighbours.cfg"  # Input File
p_state = state.setup(cfgfile)  # State setup


class TestSystem(unittest.TestCase):
    def setUp(self):
        """Setup a p_state and copy it to Clipboard"""
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
            self.assertAlmostEqual(arr[i][0], 0.0)
            self.assertAlmostEqual(arr[i][1], 0.0)
            self.assertAlmostEqual(arr[i][2], 1.0)

    def test_get_energy(self):
        # NOTE: that test is trivial
        E = system.get_energy(self.p_state)

    def test_get_energy_contributions(self):
        configuration.plus_z(self.p_state)
        configuration.domain(self.p_state, [0, 0, -1], border_cylindrical=2)
        system.update_data(self.p_state)
        E_contribs = system.get_energy_contributions(
            self.p_state, divide_by_nspins=False
        )
        E = system.get_energy(self.p_state)
        system.print_energy_array(p_state)
        self.assertEqual(len(E_contribs.values()), 3)  # There should be 3 contributions
        self.assertAlmostEqual(
            sum(E_contribs.values()), E, places=5
        )  # TODO: Apparently we can not go higher with the number of decimal places, because the order of summation differs. This Should be invesitgated.

    # NOTE: there is no way to test the system.Update_Data() and system.Print_Energy_Array()


#########


def make_suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(SystemGetters))
    return suite


if __name__ == "__main__":
    suite = make_suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete(p_state)  # delete state

    sys.exit(not success)
