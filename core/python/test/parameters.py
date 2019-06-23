import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state, parameters

import unittest

##########

#cfgfile = "core/test/input/fd_neighbours.cfg"   # Input File
cfgfile = spirit_py_dir + "/../test/input/api.cfg"   # Input File

p_state = state.setup(cfgfile)                  # State setup

class TestParameters(unittest.TestCase):
    
    def setUp(self):
        ''' Setup a p_state and copy it to Clipboard'''
        self.p_state = p_state
        
class LLG_set_get(TestParameters):
    
    # XXX: what about output functions? 
    
    def test_LLG_N_iterations(self):
        N_set = 100
        Nlog_set = 100
        parameters.llg.set_iterations(self.p_state, N_set, Nlog_set)          # try set
        N_get, Nlog_get = parameters.llg.get_iterations(self.p_state, 0, 0)   # try get
        self.assertEqual( N_set, N_get )
        self.assertEqual( Nlog_set, Nlog_get )
        
    def test_LLG_direct_minimization(self):
        parameters.llg.set_direct_minimization(self.p_state, True)      # try set
        ret = parameters.llg.get_direct_minimization(self.p_state)      # try get
        self.assertEqual( ret, True )
        parameters.llg.set_direct_minimization(self.p_state, False)     # try set
        ret = parameters.llg.get_direct_minimization(self.p_state)      # try get
        self.assertEqual( ret, False )
    
    def test_LLG_convergence(self):
        conv_set = 1.5e-3
        parameters.llg.set_convergence(self.p_state, conv_set)          # try set
        conv_get = parameters.llg.get_convergence(self.p_state)         # try get
        self.assertAlmostEqual(conv_get, conv_set)
    
    def test_LLG_timestep(self):
        dt_set = 1.5e-2
        parameters.llg.set_timestep(self.p_state, dt_set)      # try set
        dt_get = parameters.llg.get_timestep(self.p_state)     # try get
        self.assertAlmostEqual(dt_set, dt_get)
    
    def test_LLG_damping(self):
        lambda_set = 0.015
        parameters.llg.set_damping(self.p_state, lambda_set)        # try set
        lambda_get = parameters.llg.get_damping(self.p_state)       # try get
        self.assertAlmostEqual(lambda_get, lambda_set)
    
    def test_LLG_STT(self):
        use_grad_set = True
        mag_set = 0.015
        direction_set = [1., 0., 0.]        # NOTE: It will not work with tuples
        parameters.llg.set_stt(self.p_state, use_grad_set, mag_set, direction_set )     # try set
        stt_get = parameters.llg.get_stt(self.p_state)                                  # try get
        self.assertEqual(stt_get[2], use_grad_set)
        self.assertAlmostEqual(stt_get[0], mag_set)
        self.assertAlmostEqual(stt_get[1], direction_set)
    
    def test_LLG_temperature(self):
        temp_set = 100
        parameters.llg.set_temperature(self.p_state, temp_set)      # try set
        temp_get, t_gradient_inclination, t_gradient_direction = parameters.llg.get_temperature(self.p_state)     # try get
        self.assertAlmostEqual(temp_set, temp_get)

class GNEB_set_get(TestParameters):
    
    def test_GNEB_N_Iterations(self):
        N_set = 100
        Nlog_set = 100
        parameters.gneb.set_iterations(self.p_state, N_set, Nlog_set)          # try set
        N_get, Nlog_get = parameters.gneb.get_iterations(self.p_state)   # try get
        self.assertEqual( N_set, N_get )
        self.assertEqual( Nlog_set, Nlog_get )

    def test_GNEB_Convergence(self):
        conv_set = 1.5e-3
        parameters.gneb.set_convergence(self.p_state, conv_set)          # try set
        conv_get = parameters.gneb.get_convergence(self.p_state)         # try get
        self.assertAlmostEqual(conv_get, conv_set)
    
    def test_GNEB_Spring_Constant(self):
        k_set = 0.15
        parameters.gneb.set_spring_force(self.p_state, k_set)         # try set
        k_get, ratio = parameters.gneb.get_spring_force(self.p_state) # try get
        self.assertAlmostEqual(k_set, k_get)
    
    def test_GNEB_Climbing_Falling(self):
        img_type_set = 1
        parameters.gneb.set_climbing_falling(self.p_state, img_type_set)
        cf = parameters.gneb.get_climbing_falling(self.p_state)
        self.assertAlmostEqual(img_type_set, cf)
    
    def test_GNEB_Trivial_Image_Type(self):
        # NOTE: this test is trivial since we cannot get the image type
        parameters.gneb.set_image_type_automatically(self.p_state)
    
    def test_GNEB_N_Energy_Interpolations(self):
        # NOTE: this tests only the wrapping of the function since we cannot know the right value
        E_inter = parameters.gneb.get_n_energy_interpolations(self.p_state)
        self.assertTrue(E_inter > 0)
    
#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(LLG_set_get))
    suite.addTest(unittest.makeSuite(GNEB_set_get))
    return suite

if __name__ == '__main__':
    suite = suite()

    runner = unittest.TextTestRunner()
    success = runner.run(suite).wasSuccessful()

    state.delete( p_state )                         # Delete State

    sys.exit(not success)