import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import chain
from spirit import system

import unittest

##########

''' In order to avoid overhead and numerous log files in stderr in case of a 
failing test we load a p_state once and we use constructor and destructor
of each test (setUp/tearDown) to clean that p_state'''

cfgfile = spirit_py_dir + "/../test/input/api.cfg" # Input File
p_state = state.setup(cfgfile)                     # State setup
chain.image_to_clipboard( p_state )                # Copy p_state to Clipboard

class TestChain(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
    
    def tearDown(self):
        ''' clean the p_state '''
        noi = chain.get_noi( self.p_state )
        for i in range(noi-1):
            chain.pop_back( self.p_state )
        self.assertEqual( chain.get_noi( self.p_state ), 1 )

class simpleTestChain(TestChain):
    
    def test_NOI(self):
        ''' NOI -> Number of images in that chain '''
        self.assertEqual( chain.get_noi( self.p_state ), 1 )    # total 1 image
    
    def test_active_image(self):
        ''' Must return the index of the active image '''
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th

class clipboard_TestChain( TestChain ):
    
    def test_nonexisting_chain_to_clipboard(self):
        ''' Must do nothing if we want to add non existing state to clipboard'''
        chain.image_to_clipboard( self.p_state, -1, 10  );      # copy current image of 10th chain
        chain.image_to_clipboard( self.p_state, 10, 10  );      # copy 10th image 10th chain
        chain.image_to_clipboard( self.p_state, -1, -10 );      # copy current image of -10th chain

class insert_deleteTestChain(TestChain):
    
    def test_insert_after(self):
        ''' Must leave the index of active image the same '''
        chain.insert_image_after( self.p_state )                # add after active
        self.assertEqual( chain.get_noi( self.p_state ), 2 )    # total 2 images
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
    
    def test_insert_before(self):
        ''' Must reindex (increment by one) active image '''
        chain.insert_image_before( self.p_state )               # add before active
        self.assertEqual( chain.get_noi( self.p_state ), 2 )    # total 2 images
        self.assertEqual( system.get_index( self.p_state ), 1 ) # active is 1st
    
    def test_push_back(self):
        ''' Must add one image and leave active's index unchanged '''
        chain.push_back( self.p_state )                         # add after all
        self.assertEqual( chain.get_noi( self.p_state ), 2 )    # total 2 images
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
    
    def test_pop_back(self):
        ''' Must make the image with next smaller index active when the active is poped'''
        chain.insert_image_before( self.p_state )               # add before active
        self.assertEqual( system.get_index( self.p_state ), 1 ) # active is 1st
        chain.pop_back( self.p_state )                          # delete the last (active)
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
        self.assertEqual( chain.get_noi( self.p_state ), 1 )    # total 1 image

class switch_TestChain(TestChain):
    
    def test_trivial_switching(self):
        ''' Next or Prev image in image of length 1 should result to 0th active'''
        chain.next_image( self.p_state )                        # no next image
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
        chain.prev_image( self.p_state )                        # no prev image
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
    
    def test_switching(self):
        ''' For Next in the back and Prev in the front active must remain unchanged'''
        chain.insert_image_before( self.p_state )               # add before active
        self.assertEqual( system.get_index( self.p_state ), 1 ) # active is 1st
        chain.next_image( self.p_state )                        # no next image
        self.assertEqual( system.get_index( self.p_state ), 1 ) # active is 1st
        chain.prev_image( self.p_state )                        # go to prev image
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
        chain.prev_image( self.p_state )                        # no prev image
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th

class jump_TestChain(TestChain):
    
    def test_jump_trivial(self):
        ''' Must leave active image same if jump to the active'''
        chain.jump_to_image( self.p_state, idx_image=0 )        # jump to active itself
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th (the same)
    
    def test_jump(self):
        ''' Must change the active image to the one pointed by idx_image'''
        chain.insert_image_before( self.p_state )               # active is 1st
        chain.jump_to_image( self.p_state, idx_image=0 )        # jump to 0th
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
        chain.jump_to_image( self.p_state, idx_image=1 )        # jump to 1st
        self.assertEqual( system.get_index( self.p_state ), 1 ) # active is 1st
    
    def test_jump_outoflimits(self):
        '''' Must leave current active image unchanged if jump is out of chain limits'''
        chain.jump_to_image( self.p_state, idx_image=5 )        # jump to non existing
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
        chain.jump_to_image( self.p_state, idx_image=-5 )       # jump to non existing (negative)
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th

class replace_TestChain(TestChain):
    
    def test_replace_trivial(self):
        ''' Must leave active image same if replace by its self'''
        chain.replace_image( self.p_state, idx_image=0 )        # replace 0th with 0th
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
    
    def test_replace_outoflimits(self):
        ''' Must leave current image unchanged if it tries to replace an non existing image'''
        chain.replace_image( self.p_state, idx_image=5 )        # replace 0th with 5th (not exist)
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
        self.assertEqual( chain.get_noi( self.p_state ), 1 )    # total 1 image
        chain.replace_image( self.p_state, idx_image=-5 )       # replace 0th with -5th (not exist)
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
        self.assertEqual( chain.get_noi( self.p_state ), 1 )    # total 1 image

class remove_TestChain(TestChain):
    
    def test_delete_trivial(self):
        ''' Must NOT delete the image of a chain with only one image'''
        chain.delete_image( self.p_state )                      # delete 0th
        self.assertEqual( chain.get_noi( self.p_state ), 1 )    # total 1 image
    
    def test_remove_largest_index_active(self):
        ''' Must set the active to the image with the smallest index left'''
        chain.insert_image_before( self.p_state )               # active is 1st
        chain.delete_image( self.p_state )                      # delete 1st
        self.assertEqual( chain.get_noi( self.p_state ), 1 )    # total 1 image
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
    
    def test_remove_smallest_index_active(self):
        '''' Must set the active to the image with the smallest index left'''
        chain.insert_image_after( self.p_state )                # active is 0th
        chain.delete_image( self.p_state )                      # delete 0th
        self.assertEqual( chain.get_noi( self.p_state ), 1 )    # total 1 image
        self.assertEqual( system.get_index( self.p_state ), 0 ) # active is 0th
    
    def test_delete_outoflimits(self):
        
        chain.insert_image_before( self.p_state )               # active is 1st
        chain.insert_image_before( self.p_state )               # active is 2nd
        self.assertEqual( system.get_index( self.p_state ), 2 ) # active is 2nd
        self.assertEqual( chain.get_noi( self.p_state ), 3 )    # total 3 images
        
        # test the deletion of a non existing image with positive idx
        chain.delete_image( self.p_state, idx_image=5 )         # delete -5th (not exist)
        self.assertEqual( chain.get_noi( self.p_state ), 3 )    # total 3 images
        
        # test the deletion of a non existing image with negative idx
        chain.delete_image( self.p_state, idx_image=-5 )        # delete -5th (not exist)
        self.assertEqual( chain.get_noi( self.p_state ), 2 )    # total 2 images
        self.assertEqual( system.get_index( self.p_state ), 1 ) # active is 1st

# class getters_TestChain(TestChain):
    
#     # TODO: a proper way to test Rx and E values
    
#     def test_Rx(self):
#         chain.Insert_Image_Before( self.p_state )               # active is 1st
#         noi = chain.Get_NOI( self.p_state )                     # total 2 images
#         self.assertAlmostEqual( noi, 2 )
#         Rx = chain.Get_Rx( self.p_state )                       # get Rx values
#         Rx_interp = chain.Get_Rx_Interpolated( self.p_state )   # get Rx interpol 
#         self.assertNotAlmostEqual( Rx[noi-1], 0 )
#         # self.assertAlmostEqual( Rx[-1], Rx_interp[-1] )
    
#     def test_energy(self):
#         E = chain.Get_Energy( self.p_state )
#         E_interp = chain.Get_Energy_Interpolated( self.p_state )
#         print("............................")
#         print(E)
#         print(E_interp)
#         print("............................")
#         self.assertNotAlmostEqual( E[0], 0 )
#         self.assertAlmostEqual( E_interp[0], 0 )

# class data_TestChain(TestChain):
    
#     # TODO: A proper way to test Update and Setup
    
#     def test_update(self):
#         noi = chain.Get_NOI( self.p_state )
#         Ei = chain.Get_Energy( self.p_state )                   # Energy initial
#         chain.Update_Data( self.p_state )
#         Ef = chain.Get_Energy( self.p_state )                   # Energy final
#         for i in range(noi):
#             self.assertEqual( Ei[i], Ef[i] )                    # should be equal
    
#     def test_setup(self):
#         chain.Setup_Data( self.p_state )

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( simpleTestChain ) )
    suite.addTest( unittest.makeSuite( clipboard_TestChain ) )
    suite.addTest( unittest.makeSuite( insert_deleteTestChain ) )
    suite.addTest( unittest.makeSuite( switch_TestChain ) )
    suite.addTest( unittest.makeSuite( jump_TestChain ) )
    suite.addTest( unittest.makeSuite( replace_TestChain ) )
    suite.addTest( unittest.makeSuite( remove_TestChain ) )
    #suite.addTest( unittest.makeSuite( getters_TestChain ) )
    #suite.addTest( unittest.makeSuite( data_TestChain ) )
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )

    sys.exit(not success)