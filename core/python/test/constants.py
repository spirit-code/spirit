import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import constants

import unittest

#########

class TestConstants(unittest.TestCase):
    """ Units used for the next constants:
        Energy: milli-eV
        Time: pico-sec
        Magnetic field: Tesla """
    
    # TODO: Find a way to easily switch between units system (eg milli-eV to milli-Ry) and test it
    
    def test_Bohr_magnetron(self):
        self.assertEqual( constants.mu_B(), 0.057883817555 );
    
    def test_Boltzmann_Constant(self):
        self.assertEqual( constants.k_B(), 0.08617330350 )
        
    def test_Planck_constant(self):
        self.assertEqual( constants.hbar(), 0.6582119514 )
        
    def test_millirydberg(self):
        self.assertEqual( constants.mRy(), 1.0/13.605693009 )
        
    def test_gyromagnetic_ratio_of_electron(self):
        self.assertEqual( constants.gamma(), 0.1760859644 )
    
    def test_electron_g_factor(self):
        self.assertEqual( constants.g_e(), 2.00231930436182 )
        

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite(TestConstants) )
    return suite

suite = suite()

runner = unittest.TextTestRunner()
success = runner.run(suite).wasSuccessful()

sys.exit(not success)

##########