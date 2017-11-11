import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, chain, system, io, configuration

import unittest

##########

cfgfile = "core/test/input/fd_neighbours.cfg"   # Input File
io_image_test = "core/python/test/io_test_files/io_image_test"
io_chain_test = "core/python/test/io_test_files/io_chain_test"

p_state = state.setup(cfgfile)                  # State setup

class TestParameters(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
    
class Image_IO(TestParameters):
    
    def test_write(self):
        configuration.PlusZ(self.p_state)
        io.Image_Write(self.p_state, io_image_test)
    
    def test_read(self):
        nos = system.Get_NOS(self.p_state)
    
        configuration.PlusZ(self.p_state)
    
        io.Image_Write(self.p_state, io_image_test, 0, "python io test")
        io.Image_Read(self.p_state, io_image_test)
        spins = system.Get_Spin_Directions(self.p_state)
        for i in range(nos):
            self.assertAlmostEqual( spins[i][0], 0.)
            self.assertAlmostEqual( spins[i][1], 0.)
            self.assertAlmostEqual( spins[i][2], 1.)
        
        configuration.MinusZ(self.p_state)
        
        io.Image_Write(self.p_state, io_image_test, 0, "python io test")
        io.Image_Read(self.p_state, io_image_test)
        spins = system.Get_Spin_Directions(self.p_state)
        for i in range(nos):
            self.assertAlmostEqual( spins[i][0], 0.)
            self.assertAlmostEqual( spins[i][1], 0.)
            self.assertAlmostEqual( spins[i][2], -1.)
        
    def test_append(self):
        configuration.MinusZ(self.p_state)
        io.Image_Append(self.p_state, io_image_test, 0, "python io test")
        io.Image_Append(self.p_state, io_image_test, 0, "python io test")
    
class Chain_IO(TestParameters):
    
    def test_chain_write(self):
        # add two more images
        chain.Image_to_Clipboard(self.p_state)
        chain.Insert_Image_After(self.p_state)
        chain.Insert_Image_After(self.p_state)
        # set different configuration in each image
        chain.Jump_To_Image(self.p_state, 0)
        configuration.MinusZ(self.p_state)
        chain.Jump_To_Image(self.p_state, 1)
        configuration.Random(self.p_state)
        chain.Jump_To_Image(self.p_state, 2)
        configuration.PlusZ(self.p_state,)
        # write and append chain
        io.Chain_Write(self.p_state,io_chain_test, 0, "python io chain")  # this must be overwritten
        io.Chain_Write(self.p_state,io_chain_test, 0, "python io chain")
        io.Chain_Append(self.p_state,io_chain_test, 0, "python io chain")
    
#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Image_IO))
    suite.addTest(unittest.makeSuite(Chain_IO))
    return suite

suite = suite()

runner = unittest.TextTestRunner()
success = runner.run(suite).wasSuccessful()

state.delete( p_state )                         # Delete State

sys.exit(not success)