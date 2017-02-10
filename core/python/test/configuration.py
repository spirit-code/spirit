import sys

import unittest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(deine test class...))
  
    return suite


suite = suite()

runner = unittest.TextTestRunner()
success = runner.run(suite).wasSuccessful()
# man beachte das not, müsste wohl die variablen noch umbenennen
sys.exit(not success)