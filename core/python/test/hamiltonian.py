import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import hamiltonian

import unittest

##########

cfgfile = spirit_py_dir + "/../test/input/fd_neighbours.cfg"  # Input File

p_state = state.setup(cfgfile)  # State setup


class TestParameters(unittest.TestCase):
    def setUp(self):
        """Setup a p_state and copy it to Clipboard"""
        self.p_state = p_state


class Hamiltonian_set_get(TestParameters):
    def test_set_field(self):
        mag_set = 10
        dir_set = [1.0, 1.0, 0.0]
        hamiltonian.set_field(self.p_state, mag_set, dir_set)
        # TODO: test the functionality of that function when the corresponding get will be available

    def test_get_field(self):
        import math

        self.test_set_field()  # set to ensure the right values are actually set
        magnitude, normal = hamiltonian.get_field(self.p_state)
        self.assertAlmostEqual(magnitude, 10)
        expected_normal = [
            1 / math.sqrt(2),
            1 / math.sqrt(2),
            0,
        ]  # 1/sqrt(2) because of normalization
        self.assertAlmostEqual(normal[0], expected_normal[0])
        self.assertAlmostEqual(normal[1], expected_normal[1])
        self.assertAlmostEqual(normal[2], expected_normal[2])

    def test_set_anisotropy(self):
        mag_set = 0.5
        dir_set = [1.0, 1.0, 0.0]
        hamiltonian.set_anisotropy(self.p_state, mag_set, dir_set)
        # TODO: test the functionality of that function when the corresponding get will be available

    def test_set_biaxial_anisotropy(self):
        mag_set = [0.5, 0.4, 0.3, 0.2]
        dir1_set = [1.0, 1.0, 0.0]
        dir2_set = [1.0, -1.0, 0.0]
        exp_set = [[1, 0, 0], [0, 1, 1], [2, 0, 0], [1, 2, 2]]
        hamiltonian.set_biaxial_anisotropy(
            self.p_state, mag_set, exp_set, dir1_set, dir2_set
        )

    def test_get_biaxial_anisotropy(self):
        import math

        self.test_set_biaxial_anisotropy()

        n_atoms = hamiltonian.get_biaxial_anisotropy_n_atoms(self.p_state)
        n_terms = hamiltonian.get_biaxial_anisotropy_n_terms(self.p_state)

        n_terms_im = hamiltonian.get_biaxial_anisotropy_n_terms(self.p_state)
        self.assertEqual(n_terms, n_terms_im)

        self.assertTrue(n_terms >= n_atoms)

        res = hamiltonian.get_biaxial_anisotropy(self.p_state, n_atoms, n_terms)
        res_im = hamiltonian.get_biaxial_anisotropy(self.p_state)

        for mag, expected in zip(res.magnitude, res_im.magnitude, strict=True):
            self.assertAlmostEqual(mag, expected)

        for k1, k1_expected in zip(res.primary, res_im.primary, strict=True):
            for actual, expected in zip(k1, k1_expected, strict=True):
                self.assertAlmostEqual(actual, expected)

        for k2, k2_expected in zip(res.secondary, res_im.secondary, strict=True):
            for actual, expected in zip(k2, k2_expected, strict=True):
                self.assertAlmostEqual(actual, expected)

        for exponent, expected in zip(res.exponents, res_im.exponents, strict=True):
            self.assertListEqual(exponent.tolist(), expected.tolist())

        self.assertListEqual(res.site_p.tolist(), res_im.site_p.tolist())

        mag_expected = [0.5, 0.4, 0.3, 0.2]
        dir1_expected = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2), 0.0]
        dir2_expected = [1.0 / math.sqrt(2), -1.0 / math.sqrt(2), 0.0]
        exp_expected = [[1, 0, 0], [0, 1, 1], [2, 0, 0], [1, 2, 2]]

        n_terms_atom = len(mag_expected)

        for mag, expected in zip(res.magnitude, mag_expected, strict=True):
            self.assertAlmostEqual(mag, expected)

        for k1, expected in zip(res.primary[0], dir1_expected, strict=True):
            self.assertAlmostEqual(k1, expected)

        for k2, expected in zip(res.secondary[0], dir2_expected, strict=True):
            self.assertAlmostEqual(k2, expected)

        for exponent, expected in zip(res.exponents, exp_expected, strict=True):
            self.assertListEqual(exponent.tolist(), expected)

        for i in range(n_atoms):
            self.assertEqual(res.site_p[i + 1] - res.site_p[i], n_terms_atom)


#########


def make_suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(Hamiltonian_set_get))
    return suite


if __name__ == "__main__":
    suite = make_suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete(p_state)  # Delete State

    sys.exit(not success)
