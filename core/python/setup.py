import codecs
import os
import platform
import re
import shutil
import stat
import subprocess
import sys

from distutils.util import get_platform
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel as bdist_wheel_


NAME = "spirit"
PACKAGES = ['spirit']
META_PATH = os.path.join("spirit", "__init__.py")
KEYWORDS = ["Spirit", "Spin Dynamics"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: C",
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


if __name__ == "__main__":
    long_description = read('README.md')
    setup(
        name=NAME,
        description=find_meta("description"),
        long_description=long_description,
        license=find_meta("license"),
        url=find_meta("uri"),
        version=find_meta("version"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        keywords=KEYWORDS,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        package_data={
            'spirit': ['libSpirit.dylib'],
        },
    )