import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, chain, system, io, configuration, simulation

import unittest

##########

cfgfile       = spirit_py_dir + "/../test/input/fd_pairs.cfg"   # Input File
io_image_test = spirit_py_dir + "/test/io_test_files/io_image_test"
io_chain_test = spirit_py_dir + "/test/io_test_files/io_chain_test"

p_state = state.setup(cfgfile)                  # State setup

class TestParameters(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
    
class Image_IO(TestParameters):
    
    def test_write(self):
        configuration.plus_z(self.p_state)
        io.image_write(self.p_state, io_image_test)
    
    def test_read(self):
        nos = system.get_nos(self.p_state)
    
        configuration.plus_z(self.p_state)
    
        io.image_write(self.p_state, io_image_test, io.FILEFORMAT_OVF_TEXT, "python io test")
        io.image_read(self.p_state, io_image_test)
        spins = system.get_spin_directions(self.p_state)
        for i in range(nos):
            self.assertAlmostEqual( spins[i][0], 0.)
            self.assertAlmostEqual( spins[i][1], 0.)
            self.assertAlmostEqual( spins[i][2], 1.)
        
        configuration.minus_z(self.p_state)
        
        io.image_write(self.p_state, io_image_test, io.FILEFORMAT_OVF_TEXT, "python io test")
        io.image_read(self.p_state, io_image_test)
        spins = system.get_spin_directions(self.p_state)
        for i in range(nos):
            self.assertAlmostEqual( spins[i][0], 0.)
            self.assertAlmostEqual( spins[i][1], 0.)
            self.assertAlmostEqual( spins[i][2], -1.)
        
    def test_append(self):
        configuration.minus_z(self.p_state)
        io.image_append(self.p_state, io_image_test, io.FILEFORMAT_OVF_TEXT, "python io test")
        io.image_append(self.p_state, io_image_test, io.FILEFORMAT_OVF_TEXT, "python io test")

class Eigenmodes_IO(TestParameters):

    def test_write(self):
        configuration.plus_z(self.p_state)
        configuration.skyrmion(self.p_state,radius=5,phase=-90)
        simulation.start(self.p_state,"LLG","VP")
        system.update_eigenmodes(self.p_state)
        io.eigenmodes_write(self.p_state,io_image_test,io.FILEFORMAT_OVF_TEXT)

        io.image_append(self.p_state, io_image_test, io.FILEFORMAT_OVF_TEXT, "python io test")
        io.image_append(self.p_state, io_image_test, io.FILEFORMAT_OVF_TEXT, "python io test")
    
class Chain_IO(TestParameters):
    
    def test_chain_write(self):
        # add two more images
        chain.image_to_clipboard(self.p_state)
        chain.insert_image_after(self.p_state)
        chain.insert_image_after(self.p_state)
        # set different configuration in each image
        chain.jump_to_image(self.p_state, 0)
        configuration.minus_z(self.p_state)
        chain.jump_to_image(self.p_state, 1)
        configuration.random(self.p_state)
        chain.jump_to_image(self.p_state, 2)
        configuration.plus_z(self.p_state,)
        # write and append chain
        io.chain_write(self.p_state,io_chain_test, io.FILEFORMAT_OVF_TEXT, "python io chain")  # this must be overwritten
        io.chain_write(self.p_state,io_chain_test, io.FILEFORMAT_OVF_TEXT, "python io chain")
        io.chain_append(self.p_state,io_chain_test, io.FILEFORMAT_OVF_TEXT, "python io chain")
    
#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Image_IO))
    suite.addTest(unittest.makeSuite(Eigenmodes_IO))
    suite.addTest(unittest.makeSuite(Chain_IO))
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )                         # Delete State

    sys.exit(not success)
