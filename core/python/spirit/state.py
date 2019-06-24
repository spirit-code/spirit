"""
State
====================

The state contains the chain of spin systems (also called images).

To create a new state with containing a single image and run a simple example
of a **spin dynamics simulation**:

.. code-block:: python
    # Import spirit
    from spirit import state
    from spirit import simulation

    # Create a state using an input file
    with state.State("input/input.cfg") as p_state:
        # Start a LLG simulation using the SIB solver
        simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB)

or call setup and delete manually:

.. code-block:: python
    # Import spirit
    from spirit import state
    from spirit import simulation

    # Create a state using an input file
    p_state = state.setup("input/input.cfg")
    # Start a LLG simulation using the SIB solver
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB)
    # Delete the state and free memory etc.
    state.delete(p_state)

You can pass an input file specifying your initial system parameters.
If you do not pass an input file, the implemented defaults are used.
"""

import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

### State wrapper class to be used in 'with' statement
class State:
    """Wrapper Class for a Spirit State.

    Can be used as `with spirit.state.State() as p_state:`
    """

    def __init__(self, configfile="", quiet=False):
        self.p_state = setup(configfile, quiet)

    def __enter__(self):
        return self.p_state

    def __exit__(self, exc_type, exc_value, traceback):
        delete(self.p_state)

### Setup State
_State_Setup = _spirit.State_Setup
_State_Setup.argtypes = [ctypes.c_char_p, ctypes.c_bool]
_State_Setup.restype = ctypes.c_void_p
def setup(configfile="", quiet=False):
    return _State_Setup(ctypes.c_char_p(configfile.encode('utf-8')), ctypes.c_bool(quiet))

### Delete State
_State_Delete = _spirit.State_Delete
_State_Delete.argtypes = [ctypes.c_void_p]
_State_Delete.restype = None
def delete(p_state):
    return _State_Delete(ctypes.c_void_p(p_state))

### Get the date-time of the State creation
_State_To_Config = _spirit.State_To_Config
_State_To_Config.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
_State_To_Config.restype = None
def to_config(p_state, filename, comment=""):
    """Write a config (input) file corresponding to the current parameters etc. of the state."""
    _State_To_Config(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')), ctypes.c_char_p(comment.encode('utf-8')))

### Get the date-time of the State creation
_State_DateTime = _spirit.State_DateTime
_State_DateTime.argtypes = [ctypes.c_void_p]
_State_DateTime.restype = ctypes.c_char_p
def date_time(p_state):
    """Returns a string containing the date-time of the creation of the state."""
    return str(_State_DateTime(ctypes.c_void_p(p_state)))