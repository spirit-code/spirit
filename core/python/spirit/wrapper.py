# import spirit as _spirit
from spirit import configuration as _configuration
from spirit import state as _state
from spirit import system as _system
from spirit import chain as _chain
from spirit import parameters as _parameters
from spirit import io as _io
from spirit import log as _log
from spirit import simulation as _simulation
from spirit import transition as _transition
from spirit import quantities as _quantities
from spirit import htst as _htst

from enum import Enum as _Enum


class SolverKind(_Enum):
    lbfgs       = _simulation.SOLVER_LBFGS_OSO
    lbfgs_atlas = _simulation.SOLVER_LBFGS_Atlas
    vp          = _simulation.SOLVER_VP_OSO
    vp_cart     = _simulation.SOLVER_VP
    rk4         = _simulation.SOLVER_RK4
    depondt     = _simulation.SOLVER_DEPONDT
    sib         = _simulation.SOLVER_SIB
    heun        = _simulation.SOLVER_HEUN

class FileFormat(_Enum):
    ovf_bin  = _io.FILEFORMAT_OVF_BIN
    ovf_bin4 = _io.FILEFORMAT_OVF_BIN4
    ovf_bin8 = _io.FILEFORMAT_OVF_BIN8
    ovf_csv  =_io.FILEFORMAT_OVF_CSV
    ovf_text = _io.FILEFORMAT_OVF_TEXT

class LogLevel(_Enum):
    all = _log.LEVEL_ALL
    severe = _log.LEVEL_SEVERE
    error = _log.LEVEL_ERROR
    warning = _log.LEVEL_WARNING
    parameter = _log.LEVEL_PARAMETER
    info = _log.LEVEL_INFO
    debug = _log.LEVEL_DEBUG

class LogSender(_Enum):
    all = _log.SENDER_ALL
    io = _log.SENDER_IO
    gneb = _log.SENDER_GNEB
    llg = _log.SENDER_LLG
    mc = _log.SENDER_MC
    mmf = _log.SENDER_MMF
    api = _log.SENDER_API
    ui = _log.SENDER_UI


class _OutputParameters:
    _package = None

    def __init__(self, state_ptr, index=None):
        self._state_ptr = state_ptr
        self._index = index

    def set(self, options: dict):
        for key, value in options.items():
            setattr(self, key, value)
            # try:
            #     setattr(self, key, value)
            # except AttributeError:
            #     print(f'{key} is not a property of {self}')

    @property
    def output_tag(self):
        if self._index is not None:
            return self._package.get_output_tag(self._state_ptr, self._index)
        else:
            return self._package.get_output_tag(self._state_ptr)
    @output_tag.setter
    def output_tag(self, tag):
        if self._index is not None:
            self._package.set_output_tag(self._state_ptr, tag, self._index)
        else:
            self._package.set_output_tag(self._state_ptr, tag)

    @property
    def output_folder(self):
        if self._index is not None:
            return self._package.get_output_folder(self._state_ptr, self._index)
        else:
            return self._package.get_output_folder(self._state_ptr)
    @output_folder.setter
    def output_folder(self, folder: str):
        if self._index is not None:
            return self._package.set_output_folder(self._state_ptr, folder, self._index)
        else:
            return self._package.set_output_folder(self._state_ptr, folder)

    @property
    def output_any(self):
        if self._index is not None:
            return self._package.get_output_any(self._state_ptr, self._index)
        else:
            return self._package.get_output_any(self._state_ptr)
    @output_any.setter
    def output_any(self, any: bool):
        if self._index is not None:
            return self._package.set_output_any(self._state_ptr, any, self._index)
        else:
            return self._package.set_output_any(self._state_ptr, any)

    @property
    def output_initial(self):
        if self._index is not None:
            return self._package.get_output_initial(self._state_ptr, self._index)
        else:
            return self._package.get_output_initial(self._state_ptr)
    @output_initial.setter
    def output_initial(self, initial):
        if self._index is not None:
            return self._package.set_output_initial(self._state_ptr, initial, self._index)
        else:
            return self._package.set_output_initial(self._state_ptr, initial)

    @property
    def output_final(self):
        if self._index is not None:
            return self._package.get_output_final(self._state_ptr, self._index)
        else:
            return self._package.get_output_final(self._state_ptr)
    @output_final.setter
    def output_final(self, final):
        if self._index is not None:
            return self._package.set_output_final(self._state_ptr, final, self._index)
        else:
            return self._package.set_output_final(self._state_ptr, final)


class _MC_Parameters(_OutputParameters):
    def __init__(self, state_ptr, index):
        self._package = _parameters.mc
        _OutputParameters.__init__(self, state_ptr, index)

    @property
    def temperature(self):
        return _parameters.mc.get_temperature(self._state_ptr, self._index)

    @temperature.setter
    def temperature(self, value):
        _parameters.mc.set_temperature(self._state_ptr, value, self._index)


class _LLG_Parameters(_OutputParameters):
    def __init__(self, state_ptr, index):
        self._package = _parameters.llg
        _OutputParameters.__init__(self, state_ptr, index)

    @property
    def convergence(self):
        return _parameters.llg.get_convergence(self._state_ptr, self._index)
    @convergence.setter
    def convergence(self, value):
        _parameters.llg.set_convergence(self._state_ptr, value, self._index)

    @property
    def dt(self):
        return _parameters.llg.get_timestep(self._state_ptr, self._index)
    @dt.setter
    def dt(self, value):
        _parameters.llg.set_timestep(self._state_ptr, value, self._index)


class _SystemParameters:
    def __init__(self, state_ptr, index):
        self._state_ptr = state_ptr
        self._index = index

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
        self.llg._index = value

    @property
    def mc(self):
        return _MC_Parameters(self._state_ptr, self._index)

    @property
    def llg(self):
        return _LLG_Parameters(self._state_ptr, self._index)


class _SystemCalculation:
    def __init__(self, state_ptr, index=-1):
        self._state_ptr = state_ptr
        self._index = index

    def mc(self):
        pass

    def llg(self, solver: SolverKind, n_iterations=-1, step_size=-1):
        return _Simulation(self._state_ptr, _simulation.METHOD_LLG, solver.value,
            n_iterations=n_iterations, n_iterations_log=step_size, idx_image=self._index)

    def mmf(self):
        pass

class _ConfigurationSetter:
    def __init__(self, state_ptr, index=-1):
        self._state_ptr = state_ptr
        self._index = index

    def skyrmion(self, r, phase=0, order=1):
        _configuration.skyrmion(self._state_ptr, r, order=1, phase=1, up_down=False,
            achiral=False, right_left=False, pos=[0,0,0], border_rectangular=[-1,-1,-1],
            border_cylindrical=-1, border_spherical=-1, inverted=False, idx_image=self._index)

    def plus_z(self):
        _configuration.plus_z(self._state_ptr, idx_image=self._index)

class _System:
    def __init__(self, state_ptr, index=-1):
        self._state_ptr = state_ptr
        self._index = index

    @property
    def index(self):
        return self._index
    @index.setter
    def index(self, value):
        self._index = value
        self.parameters.index = value

    @property
    def directions(self):
        """Returns a `numpy.array_view` of shape (NOS, 3) with the components of each spins orientation vector.

        Changing the contents of this array_view will have direct effect on calculations etc.
        """
        # Note: because this is an array_view, it is variable even though it is a property
        return _system.get_spin_directions(self._state_ptr, self._index)

    @property
    def parameters(self):
        return _SystemParameters(self._state_ptr, self._index)

    @property
    def calculation(self):
        return _SystemCalculation(self._state_ptr, self._index)

    @property
    def set_configuration(self):
        return _ConfigurationSetter(self._state_ptr, self._index)

    def read_file(self, filename: str, idx_in_file: int):
        _io.image_read(self._state_ptr, filename, idx_image_infile=idx_in_file, idx_image_inchain=self._index)

    def write_file(self, filename: str, fileformat=FileFormat.ovf_text, comment: str=""):
        _io.image_write(self._state_ptr, filename, fileformat.value, comment, idx_image=self._index)

    def append_file(self, filename: str, fileformat=FileFormat.ovf_text, comment: str=""):
        _io.image_append(self._state_ptr, filename, fileformat.value, comment, idx_image=self._index)

    def topological_charge(self):
        return _quantities.get_topological_charge(self._state_ptr, idx_image=self._index)


class _GNEB_Parameters(_OutputParameters):
    def __init__(self, state_ptr):
        self._package = _parameters.gneb
        _OutputParameters.__init__(self, state_ptr)

    @property
    def convergence(self):
        return _parameters.gneb.get_convergence(self._state_ptr)
    @convergence.setter
    def convergence(self, value):
        _parameters.gneb.set_convergence(self._state_ptr, value)

    @property
    def spring_force(self):
        force, _ = _parameters.gneb.get_spring_force(self._state_ptr)
        return force
    @spring_force.setter
    def spring_force(self, value):
        _parameters.gneb.set_spring_force(self._state_ptr, value, self.spring_ratio)

    @property
    def spring_ratio(self):
        _, ratio = _parameters.gneb.get_spring_force(self._state_ptr)
        return ratio
    @spring_ratio.setter
    def spring_ratio(self, value):
        _parameters.gneb.set_spring_force(self._state_ptr, self.spring_force, value)


class _ChainParameters:
    def __init__(self, state_ptr):
        self._state_ptr = state_ptr

    @property
    def gneb(self):
        return _GNEB_Parameters(self._state_ptr)


class _HtstResult:
    def __init__(self, state_ptr):
        self._state_ptr = state_ptr
        self._info_dict = _htst.get_info_dict(self._state_ptr)

    @property
    def temperature_exponent(self):
        return self._info_dict['temperature_exponent']

    @property
    def me(self):
        return self._info_dict['me']

    @property
    def Omega_0(self):
        return self._info_dict['Omega_0']

    @property
    def s(self):
        return self._info_dict['s']

    @property
    def volume_min(self):
        return self._info_dict['volume_min']

    @property
    def volume_sp(self):
        return self._info_dict['volume_sp']

    @property
    def prefactor_dynamical(self):
        return self._info_dict['prefactor_dynamical']

    @property
    def prefactor(self):
        return self._info_dict['prefactor']

    @property
    def n_eigenmodes_keep(self):
        return self._info_dict['n_eigenmodes_keep']

    @property
    def eigenvalues_min(self):
        return _htst.get_eigenvalues_min(self._state_ptr)

    @property
    def eigenvectors_min(self):
        return _htst.get_eigenvectors_min(self._state_ptr)

    @property
    def eigenvalues_sp(self):
        return _htst.get_eigenvalues_sp(self._state_ptr)

    @property
    def eigenvectors_sp(self):
        return _htst.get_eigenvectors_sp(self._state_ptr)

    @property
    def velocities(self):
        return _htst.get_velocities(self._state_ptr)


class _ChainCalculation:
    def __init__(self, state_ptr):
        self._state_ptr = state_ptr

    def gneb(self, solver: SolverKind, n_iterations=-1, step_size=-1):
        return _Simulation(self._state_ptr, _simulation.METHOD_GNEB, solver.value,
            n_iterations=n_iterations, n_iterations_log=step_size)

    def htst(self, idx_image_minimum, idx_image_sp, n_eigenmodes_keep=-1):
        _htst.calculate(self._state_ptr, idx_image_minimum, idx_image_sp, n_eigenmodes_keep)
        return _HtstResult(self._state_ptr)


class _Chain:
    def __init__(self, state_ptr):
        self._state_ptr = state_ptr
        noi = _chain.get_noi(self._state_ptr)
        self._images = [_System(state_ptr, idx) for idx in range(noi)]

    # Basics

    def resize(self, length):
        _chain.image_to_clipboard(self._state_ptr)
        _chain.set_length(self._state_ptr, length)
        if length < len(self._images):
            for idx in range(length, len(self._images)):
                del self._images[idx]
        if length > len(self._images):
            for idx in range(len(self._images), length):
                self._images.append(_System(self._state_ptr, idx))

    # I/O

    def read_file(self, filename: str, starting_image=0, ending_image=-1, insert_idx=0):
        _io.chain_read(self._state_ptr, filename, starting_image, ending_image, insert_idx)

    def write_file(self, filename: str, fileformat=FileFormat.ovf_text, comment: str=""):
        _io.chain_write(self._state_ptr, filename, fileformat.value, comment)

    def append_file(self, filename: str, fileformat=FileFormat.ovf_text, comment: str=""):
        _io.chain_append(self._state_ptr, filename, fileformat.value, comment)

    # Transitions

    def interpolate(self, idx_1=0, idx_2=-1):
        idx_1 = self._images.index(self._images[idx_1])
        idx_2 = self._images.index(self._images[idx_2])
        _transition.homogeneous(self._state_ptr, idx_1, idx_2)

    # List emulation functions

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        if index < len(self._images):
            return self._images[index]
        else:
            raise IndexError

    def __setitem__(self, index, image):
        _chain.image_to_clipboard(self._state_ptr, image.index)
        _chain.replace_image(self._state_ptr, index)

    def __delitem__(self, index):
        _chain.delete_image(self._state_ptr, index)
        if index >= len(self._images):
            raise IndexError
        for image in self._images[index:]:
            image.index -= 1
        del self._images[index]

    # TODO
    # def remove(self, index):
    #     chain.delete_image(self._state_ptr, index)
    #     if index >= len(self._images):
    #         return
    #     for image in self._images[index:]:
    #         image._index -= 1
    #     del self._images[index]

    # def __missing__(self, index):
    #     pass

    def __iter__(self):
        for image in self._images:
            yield image

    # TODO for concatenations etc
    # __add__(), __radd__(), __iadd__(), __mul__(), __rmul__() and __imul__()

    # TODO for `in` operator
    # def __contains__(self, index):
    #     pass

    def insert(self, index, image):
        _chain.image_to_clipboard(self._state_ptr, image._index)
        _chain.insert_image_after(self._state_ptr, index-1)
        self._images.insert(index, _System(self._state_ptr, index))
        for image in self._images[index:]:
            image._index += 1

    def append(self, image):
        self.insert(len(self._images), image)


class _Log:
    def __init__(self, state_ptr):
        self._state_ptr = state_ptr

    def __call__(self, message, level=LogLevel.all, sender=LogSender.all, idx_image=-1):
        _log.send(self._state_ptr, level.value, sender.value, message, idx_image)

    def append(self):
        _log.append(self._state_ptr)

    @property
    def n_entries(self):
        return _log.get_n_entries(self._state_ptr)
    @property
    def n_errors(self):
        return _log.get_n_errors(self._state_ptr)
    @property
    def n_warnings(self):
        return _log.get_n_warnings(self._state_ptr)

    @property
    def output_file_tag(self):
        # return _log.get_output_file_tag(self._state_ptr)
        return "" # TODO
    @output_file_tag.setter
    def output_file_tag(self, value):
        _log.set_output_file_tag(self._state_ptr, value)

    @property
    def output_folder(self):
        # return _log.get_output_folder(self._state_ptr)
        return "" # TODO
    @output_folder.setter
    def output_folder(self, value):
        _log.set_output_folder(self._state_ptr, value)

    @property
    def output_to_console(self):
        return _log.get_output_to_console(self._state_ptr)
    @output_to_console.setter
    def output_to_console(self, value):
        _log.set_output_to_console(self._state_ptr, value, self.console_level)

    @property
    def console_level(self):
        return _log.get_output_console_level(self._state_ptr)
    @console_level.setter
    def console_level(self, level):
        # print('setting log level ', level.value)
        _log.set_output_to_console(self._state_ptr, self.output_to_console, level.value)
    # def set_console_level(self, level):
    #     print('setting log level ', level.value)
    #     _log.set_output_to_console(self._state_ptr, self.output_to_console, level.value)

    @property
    def output_to_file(self):
        return _log.get_output_to_file(self._state_ptr)
    @output_to_file.setter
    def output_to_file(self, value):
        _log.set_output_to_console(self._state_ptr, value, self.file_level)

    @property
    def file_level(self):
        return _log.get_output_file_level(self._state_ptr)
    @file_level.setter
    def file_level(self, level):
        _log.set_output_to_file(self._state_ptr, self.output_to_file, level.value)


class _Simulation:
    def __init__(self, state_ptr, method_type: str, solver_type=None, n_iterations=-1, n_iterations_log=-1, idx_image=-1):
        self._state_ptr = state_ptr
        self._method_type = method_type
        self._solver_type = solver_type
        self._n_iterations = n_iterations
        self._n_iterations_log = n_iterations_log
        self._index = idx_image

    def start(self):
        _simulation.start(self._state_ptr, self._method_type, self._solver_type, self._n_iterations,
            self._n_iterations_log, single_shot=False, idx_image=self._index)

    def stop(self):
        _simulation.stop(self._state_ptr, self._index)

    def __call__(self):
        self.start()

    def __iter__(self):
        """Usage examples:

            for i_iter in sim:
                print(f'iteration {i_iter}')

            i_iter=0
            while i_iter < 1000:
                i_iter = next(sim)
        """
        _simulation.start(self._state_ptr, self._method_type, self._solver_type, self._n_iterations,
            self._n_iterations_log, single_shot=True, idx_image=self._index)
        try:
            if self._n_iterations < 0:
                i = 0
                while True:
                    _simulation.single_shot(self._state_ptr, self._index)
                    i += 1
                    yield i
            else:
                for i in range(self._n_iterations):
                    _simulation.single_shot(self._state_ptr, self._index)
                    yield i+1
        except GeneratorExit:
            pass
        _simulation.stop(self._state_ptr, self._index)


class State:
    """Wrapper Class for a Spirit state.
    """

    def __init__(self, configfile:str="", quiet:bool=False):
        self._ptr = _state.setup(configfile, quiet)

    def __del__(self):
        _state.delete(self._ptr)

    @property
    def active_image(self):
        # Due to index=-1, this will always point to the active image
        return _System(self._ptr)

    def jump_to_image(self, index:int):
        _chain.jump_to_image(self._ptr, index)

    def to_config(self, filename:str, comment:str=""):
        _state.to_config(self._ptr, filename, comment)

    @property
    def date_time(self):
        return _state.date_time(self._ptr)

    @property
    def systems(self):
        return _Chain(self._ptr)

    @property
    def log(self):
        return _Log(self._ptr)

    @property
    def parameters(self):
        return _ChainParameters(self._ptr)

    @property
    def calculation(self):
        return _ChainCalculation(self._ptr)