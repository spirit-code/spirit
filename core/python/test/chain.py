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

cfgfile = "input/input.cfg"                 # Input File
p_state = state.setup(cfgfile)              # State setup
chain.Image_to_Clipboard( p_state )         # Copy p_state to Clipboard

class TestChain(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
    
    def tearDown(self):
        ''' clean the p_state '''
        noi = chain.Get_NOI( self.p_state )
        for i in range(noi-1):
            chain.Pop_Back( self.p_state )
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )

class simpleTestChain(TestChain):
    
    def test_chain_index(self):
        ''' Must be always 0 since we have only one chain'''
        self.assertEqual( chain.Get_Index( self.p_state ), 0 )  # 0th chain
    
    def test_NOI(self):
        ''' NOI -> Number of images in that chain '''
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )    # total 1 image
    
    def test_active_image(self):
        ''' Must return the index of the active image '''
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th

class insert_deleteTestChain(TestChain):
    
    def test_insert_after(self):
        ''' Must leave the index of active image the same '''
        chain.Insert_Image_After( self.p_state )                # add after active
        self.assertEqual( chain.Get_NOI( self.p_state ), 2 )    # total 2 images
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
    
    def test_insert_before(self):
        ''' Must reindex (increment by one) active image '''
        chain.Insert_Image_Before( self.p_state )               # add before active
        self.assertEqual( chain.Get_NOI( self.p_state ), 2 )    # total 2 images
        self.assertEqual( system.Get_Index( self.p_state ), 1 ) # active is 1st
    
    def test_push_back(self):
        ''' Must add one image and leave active's index unchanged '''
        chain.Push_Back( self.p_state )                         # add after all
        self.assertEqual( chain.Get_NOI( self.p_state ), 2 )    # total 2 images
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
    
    def test_pop_back(self):
        ''' Must make the image with next smaller index active when the active is poped'''
        chain.Insert_Image_Before( self.p_state )               # add before active
        self.assertEqual( system.Get_Index( self.p_state ), 1 ) # active is 1st
        chain.Pop_Back( self.p_state )                          # delete the last (active)
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )    # total 1 image

class switch_TestChain(TestChain):
    
    def test_trivial_switching(self):
        ''' Next or Prev image in image of length 1 should result to 0th active'''
        chain.Next_Image( self.p_state )                        # no next image
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
        chain.Prev_Image( self.p_state )                        # no prev image
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
    
    def test_switching(self):
        ''' For Next in the back and Prev in the front active must remain unchanged'''
        chain.Insert_Image_Before( self.p_state )               # add before active
        self.assertEqual( system.Get_Index( self.p_state ), 1 ) # active is 1st
        chain.Next_Image( self.p_state )                        # no next image
        self.assertEqual( system.Get_Index( self.p_state ), 1 ) # active is 1st
        chain.Prev_Image( self.p_state )                        # go to prev image
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
        chain.Prev_Image( self.p_state )                        # no prev image
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th

class jump_TestChain(TestChain):
    
    def test_jump_trivial(self):
        ''' Must leave active image same if jump to the active'''
        chain.Jump_To_Image( self.p_state, idx_image=0 )        # jump to active itself
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th (the same)
    
    def test_jump(self):
        ''' Must change the active image to the one pointed by idx_image'''
        chain.Insert_Image_Before( self.p_state )               # active is 1st
        chain.Jump_To_Image( self.p_state, idx_image=0 )        # jump to 0th
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
        chain.Jump_To_Image( self.p_state, idx_image=1 )        # jump to 1st
        self.assertEqual( system.Get_Index( self.p_state ), 1 ) # active is 1st
    
    def test_jump_outoflimits(self):
        '''' Must leave current active image unchanged if jump is out of chain limits'''
        chain.Jump_To_Image( self.p_state, idx_image=5 )        # jump to non existing
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
        chain.Jump_To_Image( self.p_state, idx_image=-5 )       # jump to non existing (negative)
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th

class replace_TestChain(TestChain):
    
    def test_replace_trivial(self):
        ''' Must leave active image same if replace by its self'''
        chain.Replace_Image( self.p_state, idx_image=0 )        # replace 0th with 0th
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
    
    def test_replace_outoflimits(self):
        ''' Must leave current image unchanged if it tries to replace an non existing image'''
        chain.Replace_Image( self.p_state, idx_image=5 )        # replace 0th with 5th (not exist)
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )    # total 1 image
        chain.Replace_Image( self.p_state, idx_image=-5 )       # replace 0th with -5th (not exist)
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )    # total 1 image

class remove_TestChain(TestChain):
    
    def test_delete_trivial(self):
        ''' Must NOT delete the image of a chain with only one image'''
        chain.Delete_Image( self.p_state )                      # delete 0th
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )    # total 1 image
    
    def test_remove_largest_index_active(self):
        ''' Must set the active to the image with the smallest index left'''
        chain.Insert_Image_Before( self.p_state )               # active is 1st
        chain.Delete_Image( self.p_state )                      # delete 1st
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )    # total 1 image
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
    
    def test_remove_smallest_index_active(self):
        '''' Must set the active to the image with the smallest index left'''
        chain.Insert_Image_After( self.p_state )                # active is 0th
        chain.Delete_Image( self.p_state )                      # delete 0th
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )    # total 1 image
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 0th
    
    def test_delete_outoflimits(self):
        
        # BUG: delete image out of bound should NOT reduce the number of images NOI
        
        chain.Insert_Image_Before( self.p_state )               # active is 1st
        chain.Insert_Image_Before( self.p_state )               # active is 2nd
        self.assertEqual( chain.Get_NOI( self.p_state ), 3 )    # total 1 image
        chain.Delete_Image( self.p_state, idx_image=5 )         # delete 5th (not exist)
        self.assertEqual( chain.Get_NOI( self.p_state ), 2 )    # total 1 image
        chain.Delete_Image( self.p_state, idx_image=-5 )        # delete -5th (not exist)
        self.assertEqual( chain.Get_NOI( self.p_state ), 1 )    # total 1 image
        self.assertEqual( system.Get_Index( self.p_state ), 0 ) # active is 1st

class getters_TestChain(TestChain):
    
    # TODO: a proper way to test Rx and E values
    
    def test_Rx(self):
        chain.Insert_Image_Before( self.p_state )               # active is 1st
        noi = chain.Get_NOI( self.p_state )                     # total 2 images
        Rx = chain.Get_Rx( self.p_state )                       # get Rx values
        Rx_interp = chain.Get_Rx_Interpolated( self.p_state )   # get Rx interpol 
        self.assertNotAlmostEqual( Rx[i], 0 )
        self.assertAlmostEqual( len(Rx_interp), 0 )
    
    def test_energy(self):
        E = chain.Get_Energy( self.p_state )
        E_interp = chain.Get_Energy_Interpolated( self.p_state )
        self.assertNotAlmostEqual( E[0], 0 )
        self.assertAlmostEqual( E_interp[0], 0 )

class data_TestChain(TestChain):
    
    # TODO: A proper way to test Update and Setup
    
    def test_update(self):
        Ei = chain.Get_Energy( self.p_state )                   # Energy initial
        chain.Update_Data( self.p_state )
        Ef = chain.Get_Energy( self.p_state )                   # Energy final
        self.assertEqual( Ei, Ef )                              # should be equal
    
    def test_setup(self):
        chain.Setup_Data( self.p_state )

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( simpleTestChain ) )
    suite.addTest( unittest.makeSuite( insert_deleteTestChain ) )
    suite.addTest( unittest.makeSuite( switch_TestChain ) )
    suite.addTest( unittest.makeSuite( jump_TestChain ) )
    suite.addTest( unittest.makeSuite( replace_TestChain ) )
    suite.addTest( unittest.makeSuite( remove_TestChain ) )
    #suite.addTest( unittest.makeSuite( getters_TestChain ) )
    #suite.addTest( unittest.makeSuite( data_TestChain ) )
    return suite

suite = suite()

runner = unittest.TextTestRunner()
success = runner.run(suite).wasSuccessful()

state.delete( p_state )

sys.exit(not success)