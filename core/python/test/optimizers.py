import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import system
from spirit import simulation
from spirit import geometry
from spirit import configuration
from spirit import quantities

import unittest

#########

cfgfile = "input/optimizers/test.cfg"       # Input File

# We have implemented 4 optimizers ( "SIB", "SIB2", "VP", "Heun" )
optimizers    = [ "SIB", "SIB2", "VP" ]         # XXX: Omit Heun optimizer 
energy        = [ .0, .0, .0, .0 ]
magnetization = [ []*3, []*3, []*3, []*3 ]

class TestOptimizer( unittest.TestCase ):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

class test_Optimizers_LLG( TestOptimizer ):
    # TODO: Find DRY solution to avoid overhead of creating and deleting states 
    
    def setUp(self):
        ''' Setup a p_state '''
        self.p_state = state.setup( cfgfile )       # steate setup
        self.center = [ .0, .0, .0 ]
        geometry.Get_Center( self.p_state, self.center )
    
    def tearDown(self):
        ''' delete p_state '''
        state.delete( self.p_state )
    
    def test_sib(self):
        geometry.Get_Center( self.p_state, self.center )
        configuration.Skyrmion( self.p_state, 5, phase=-90, pos=self.center )
        simulation.PlayPause( self.p_state, "LLG", "SIB" )
        energy[0] = system.Get_Energy( self.p_state )
        magnetization[0] = quantities.Get_Magnetization( self.p_state )
    
    def test_sib2(self):
        geometry.Get_Center( self.p_state, self.center )
        configuration.Skyrmion( self.p_state, 5, phase=-90, pos=self.center )
        simulation.PlayPause( self.p_state, "LLG", "SIB2" )
        energy[1] = system.Get_Energy( self.p_state )
        magnetization[1] = quantities.Get_Magnetization( self.p_state )
    
    def test_vp(self):
        geometry.Get_Center( self.p_state, self.center )
        configuration.Skyrmion( self.p_state, 5, phase=-90, pos=self.center )
        simulation.PlayPause( self.p_state, "LLG", "VP" )
        energy[2] = system.Get_Energy( self.p_state )
        magnetization[2] = quantities.Get_Magnetization( self.p_state )        
    
    # XXX: Heun optimizer needs almost 25[sec] making testing really slow
    # def test_heun(self):
    #     geometry.Get_Center( self.p_state, self.center )
    #     configuration.Skyrmion( self.p_state, 5, phase=-90, pos=self.center )
    #     simulation.PlayPause( self.p_state, "LLG", "Heun" )
    #     energy[3] = system.Get_Energy( self.p_state )
    #     magnetization[3] = quantities.Get_Magnetization( self.p_state )

class test_Energies( TestOptimizer ):
    
    def test_Foo(self):
        # for every optimizer run
        for i in range( len(optimizers) ):
            self.assertAlmostEqual( magnetization[i][0], 0 )    # x-direction should be zero
            self.assertAlmostEqual( magnetization[i][1], 0 )    # y-direction should be zero
            print( energy[i] )
            # for every other combination
            for j in range(i+1, len(optimizers) ):
                self.assertAlmostEqual( energy[i], energy[j] )      # equal magnetization
                self.assertAlmostEqual( magnetization[i][2], magnetization[j][2] ) # equal energy

#########

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( test_Optimizers_LLG ) )
    suite.addTest( unittest.makeSuite( test_Energies ) )
    return suite

suite = suite()

runner = unittest.TextTestRunner()
success = runner.run(suite).wasSuccessful()

sys.exit(not success)