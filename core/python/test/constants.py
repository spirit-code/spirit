import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import constants
from spirit.scalar import scalar

import numpy as np
import unittest

#########

class TestConstants(unittest.TestCase):
    """ Units used for the next constants:
        Energy: milli-eV
        Time: pico-sec
        Magnetic field: Tesla """
    
    # TODO: Find a way to easily switch between units system (eg milli-eV to milli-Ry) and test it

    def test_Bohr_magneton(self):
        self.assertEqual( scalar(constants.mu_B).value, scalar(0.057883817555).value )
    
    def test_Boltzmann_Constant(self):
        self.assertEqual( scalar(constants.k_B).value, scalar(0.08617330350).value )
        
    def test_Planck_constant(self):
        self.assertEqual( scalar(constants.hbar).value, scalar(0.6582119514).value )
        
    def test_millirydberg(self):
        self.assertEqual( scalar(constants.mRy).value, scalar(1.0/13.605693009).value )
        
    def test_gyromagnetic_ratio_of_electron(self):
        self.assertEqual( scalar(constants.gamma).value, scalar(0.1760859644).value )
    
    def test_electron_g_factor(self):
        self.assertEqual( scalar(constants.g_e).value, scalar(2.00231930436182).value )
        

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite(TestConstants) )
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    sys.exit(not success)