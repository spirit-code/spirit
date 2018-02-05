import os
import re
import sys
import platform
import subprocess
import datetime

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel as bdist_wheel_
from distutils.util import get_platform
from distutils.version import LooseVersion


def get_git_commit_datetime():
    try:
        commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        commit_datetime = subprocess.check_output("git show -s --format=%ci "+commit_hash, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        print(commit_datetime)
        datetime_object = datetime.datetime.strptime(commit_datetime, '%Y-%m-%d %H:%M:%S +%f')
        print("{:%Y%m%d%H%M%S}".format(datetime_object))
        return "{:%Y%m%d%H%M%S}".format(datetime_object)
    except subprocess.CalledProcessError as cpe:
        print(cpe.output)
        return "00000000000000"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DBUILD_DEMO=OFF',
                      '-DBUILD_PYTHON_BINDINGS=ON',
                      '-DMODULE_DEV_TAG='+version_extension]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir]  + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


class bdist_wheel(bdist_wheel_):
    def finalize_options(self):
        from sys import platform as _platform
        platform_name = get_platform()
        if _platform == "linux" or _platform == "linux2":
            # Linux
            platform_name = 'manylinux1_x86_64'
        
        bdist_wheel_.finalize_options(self)
        self.universal = True
        self.plat_name_supplied = True
        self.plat_name = platform_name


# Read version from file
vfrendering_version = open('Version.txt').read().strip()


# Set extra version tag for dev builds if corresponding env variable is set
version_extension = ""
add_version_extension = os.environ.get("VFRENDERING_ADD_VERSION_EXTENSION", "")
if add_version_extension.lower() in ("yes", "true", "t", "1"):
    timepoint_string = get_git_commit_datetime()
    if timepoint_string == "00000000000000":
        timepoint_string = "{:%Y%m%d%H%M}".format(datetime.datetime.now())
    version_extension = ".dev" + timepoint_string
    print("setup.py: package version suffix = ", version_extension)


# Setup configuration
setup(
    name='pyVFRendering',
    version=vfrendering_version + version_extension,
    author='Florian Rhiem',
    author_email='f.rhiem@fz-juelich.de',
    url='https://github.com/FlorianRhiem/VFRendering',
    description='VFRendering python bindings',
    long_description='',
    ext_modules=[CMakeExtension('pyVFRendering')],
    cmdclass = {'build_ext': CMakeBuild, 'bdist_wheel': bdist_wheel},
)