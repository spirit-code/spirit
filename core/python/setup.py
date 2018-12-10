import codecs
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import datetime

from distutils.util import get_platform
from setuptools import setup, Command
from wheel.bdist_wheel import bdist_wheel as bdist_wheel_


NAME = "spirit"
PACKAGES = ['spirit', 'spirit.parameters']
META_PATH = os.path.join("spirit", "__init__.py")
KEYWORDS = ["Spirit", "Spin Dynamics"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: C",
    "Programming Language :: C++",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
INSTALL_REQUIRES = ["numpy"]

###############################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))

def get_git_commit_datetime():
    try:
        commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        commit_datetime = subprocess.check_output("git show -s --format=%ci "+commit_hash, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        print(commit_datetime)
        commit_datetime = ' '.join(commit_datetime.split()[:-1])
        print(commit_datetime)
        datetime_object = datetime.datetime.strptime(commit_datetime, '%Y-%m-%d %H:%M:%S')
        print("{:%Y%m%d%H%M%S}".format(datetime_object))
        return "{:%Y%m%d%H%M%S}".format(datetime_object)
    except subprocess.CalledProcessError as cpe:
        print(cpe.output)
        return "00000000000000"

import unittest
def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test', pattern='*.py')
    return test_suite

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


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


if __name__ == "__main__":
    # If the environment variable SPIRIT_VERSION_SUFFIX is defined,
    # it is appended to the package version number.
    version_suffix = ""
    add_version_suffix = os.environ.get("SPIRIT_ADD_VERSION_SUFFIX", "")
    if add_version_suffix.lower() in ("yes", "true", "t", "1"):
        timepoint_string = get_git_commit_datetime()
        if timepoint_string == "00000000000000":
            timepoint_string = "{:%Y%m%d%H%M}".format(datetime.datetime.now())
        version_suffix = ".dev"+timepoint_string
        print("setup.py: package version suffix = ", version_suffix)

    # Setup the package info
    setup(
        name             = NAME,
        description      = find_meta("description"),
        long_description = read('README.md'),
        license          = find_meta("license"),
        url              = find_meta("uri"),
        version          = find_meta("version")+version_suffix,
        author           = find_meta("author"),
        author_email     = find_meta("email"),
        maintainer       = find_meta("author"),
        maintainer_email = find_meta("email"),
        keywords         = KEYWORDS,
        packages         = PACKAGES,
        classifiers      = CLASSIFIERS,
        install_requires = INSTALL_REQUIRES,
        package_data     = {
            'spirit': ['libSpirit.dylib', 'libSpirit.so', 'libSpirit.dll'],
        },
        cmdclass         = {'bdist_wheel': bdist_wheel, 'clean': CleanCommand},
        test_suite       = 'setup.my_test_suite',
    )