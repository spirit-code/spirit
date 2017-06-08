import os
import sys

# spirit_py_dir = os.path.dirname(os.path.realpath(__file__))
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, spirit_py_dir)

from spirit import state
from spirit import transition

import unittest

##########