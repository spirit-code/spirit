import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import chain

import unittest

##########

cfgfile = "input/input.cfg"

class TestChain(unittest.TestCase):
    def test(self):
        with state.State(cfgfile) as p_state:
            # Inserting
            chain.Image_to_Clipboard(p_state)
            chain.Insert_Image_Before(p_state)
            chain.Insert_Image_After(p_state)
            chain.Chain_Push_Back(p_state)
            self.assertEqual(chain.Get_NOI(p_state), 4)
            self.assertEqual(chain.Get_Index(p_state), 1)
            # Switch
            chain.Next_Image(p_state)
            chain.Prev_Image(p_state)
            self.assertEqual(chain.Get_Index(p_state), 1)
            # Jump
            chain.Jump_To_Image(p_state, idx_image=0)
            self.assertEqual(chain.Get_Index(p_state), 0)
            # Replacing and removing
            chain.Replace_Image(p_state)
            chain.Delete_Image(p_state)
            chain.Pop_Back(p_state)
            # Setup & Update Data
            chain.Setup_Data(p_state)
            chain.Update_Data(p_state)
            # Getters
            Rx = chain.Get_Rx(p_state)
            Rx_interpolated = chain.Get_Rx_Interpolated(p_state)
            E = chain.Get_Energy(p_state)
            E_interpolated = chain.Get_Energy_Interpolated(p_state)

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestChain))
  
    return suite


suite = suite()

runner = unittest.TextTestRunner()
success = runner.run(suite).wasSuccessful()
sys.exit(not success)