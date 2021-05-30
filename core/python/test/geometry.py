import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, geometry

import unittest

##########

cfgfile = spirit_py_dir + "/../test/input/fd_neighbours.cfg"   # Input File

p_state = state.setup(cfgfile)                  # State setup

class TestParameters(unittest.TestCase):

    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state

class Geometry(TestParameters):

    def test_bounds(self):
        minb, maxb = geometry.get_bounds(self.p_state)
        # From the api.cfg the space is 2:2:1 particles
        self.assertEqual(minb[0], 0)
        self.assertEqual(minb[1], 0)
        self.assertEqual(minb[2], 0)
        self.assertEqual(maxb[0], 1)
        self.assertEqual(maxb[1], 1)
        self.assertEqual(maxb[2], 0)

    def test_center(self):
        center = geometry.get_center(self.p_state)
        # From the api.cfg the space is 2:2:1 particles
        self.assertEqual(center[0], 0.5)
        self.assertEqual(center[1], 0.5)
        self.assertEqual(center[2], 0)

    def test_bravais_vector(self):
        a, b, c = geometry.get_bravais_vectors(self.p_state)
        # From the api.cfg the bravais vectors are (1,0,0), (0,1,0), (0,0,1)
        self.assertEqual(a[0], b[1])
        self.assertEqual(b[1], c[2])
        # Check also that the bravais lattice type matches simple cubic
        lattice_type = geometry.get_bravais_lattice_type(self.p_state)
        self.assertEqual(lattice_type, geometry.BRAVAIS_LATTICE_SC)

    def test_N_cells(self):
        ncells = geometry.get_n_cells(self.p_state)
        self.assertEqual(ncells[0], 2)
        self.assertEqual(ncells[1], 2)
        self.assertEqual(ncells[2], 1)

    def test_dimensionality(self):
        dim = geometry.get_dimensionality(self.p_state)
        self.assertEqual(dim, 2)

    def test_positions(self):
        positions = geometry.get_positions(self.p_state)
        # spin at (0,0,0)
        self.assertAlmostEqual(positions[0][0], 0)
        self.assertAlmostEqual(positions[0][1], 0)
        self.assertAlmostEqual(positions[0][2], 0)
        # spin at (1,0,0)
        self.assertAlmostEqual(positions[1][0], 1)
        self.assertAlmostEqual(positions[1][1], 0)
        self.assertAlmostEqual(positions[1][2], 0)
        # spin at (0,1,0)
        self.assertAlmostEqual(positions[2][0], 0)
        self.assertAlmostEqual(positions[2][1], 1)
        self.assertAlmostEqual(positions[2][2], 0)
        # spin at (1,1,0)
        self.assertAlmostEqual(positions[3][0], 1)
        self.assertAlmostEqual(positions[3][1], 1)
        self.assertAlmostEqual(positions[3][2], 0)

    def test_atom_types(self):
        types = geometry.get_atom_types(self.p_state)
        self.assertEqual(len(types), 4)
        self.assertEqual(types[0], 0)
        self.assertEqual(types[1], 0)
        self.assertEqual(types[2], 0)
        self.assertEqual(types[3], 0)

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Geometry))
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )                         # Delete State

    sys.exit(not success)