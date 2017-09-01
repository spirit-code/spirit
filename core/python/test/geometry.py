import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import geometry

import unittest

##########

cfgfile = "core/test/input/fd_neighbours.cfg"   # Input File

p_state = state.setup(cfgfile)                  # State setup

class TestParameters(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
        
class Geometry(TestParameters):
    
    def test_bounds(self):
        minb, maxb = geometry.Get_Bounds(self.p_state)
        # From the api.cfg the space is 2:2:1 particles
        self.assertEqual(minb[0], 0)
        self.assertEqual(minb[1], 0)
        self.assertEqual(minb[2], 0)
        self.assertEqual(maxb[0], 1)
        self.assertEqual(maxb[1], 1)
        self.assertEqual(maxb[2], 0)
    
    def test_center(self):
        center = geometry.Get_Center(self.p_state)
        # From the api.cfg the space is 2:2:1 particles
        self.assertEqual(center[0], 0.5)
        self.assertEqual(center[1], 0.5)
        self.assertEqual(center[2], 0)
    
    def test_basis_vector(self):
        a, b, c = geometry.Get_Basis_Vectors(self.p_state)
        # From the api.cfg the basis is (1,0,0), (0,1,0), (0,0,1)
        self.assertEqual(a[0], b[1], c[2])
    
    def test_N_cells(self):
        ncells = geometry.Get_N_Cells(self.p_state)
        self.assertEqual(ncells[0], 2)
        self.assertEqual(ncells[1], 2)
        self.assertEqual(ncells[2], 1)
        
    def test_translational_vector(self):
        ta, tb, tc = geometry.Get_Translation_Vectors(self.p_state)
        # From the api.cfg the tvec are (1,0,0), (0,1,0), (0,0,1)
        self.assertEqual(ta[0], tb[1], tc[2])
    
    def test_dimensionality(self):
        dimen = geometry.Get_Dimensionality(self.p_state)
        self.assertEqual(dimen, 2)
    
    def test_spin_positions(self):
        spin_positions = geometry.Get_Spin_Positions(self.p_state)
        # spin at (0,0,0)
        self.assertAlmostEqual(spin_positions[0][0], 0)
        self.assertAlmostEqual(spin_positions[0][1], 0)
        self.assertAlmostEqual(spin_positions[0][2], 0)
        # spin at (1,0,0)
        self.assertAlmostEqual(spin_positions[1][0], 1)
        self.assertAlmostEqual(spin_positions[1][1], 0)
        self.assertAlmostEqual(spin_positions[1][2], 0)
        # spin at (0,1,0)
        self.assertAlmostEqual(spin_positions[2][0], 0)
        self.assertAlmostEqual(spin_positions[2][1], 1)
        self.assertAlmostEqual(spin_positions[2][2], 0)
        # spin at (1,1,0)
        self.assertAlmostEqual(spin_positions[3][0], 1)
        self.assertAlmostEqual(spin_positions[3][1], 1)
        self.assertAlmostEqual(spin_positions[3][2], 0)
    
    def test_atom_types(self):
        types = geometry.Get_Atom_Types(self.p_state)
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

suite = suite()

runner = unittest.TextTestRunner()
success = runner.run(suite).wasSuccessful()

state.delete( p_state )                         # Delete State

sys.exit(not success)