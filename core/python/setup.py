import codecs
import datetime
import os
import re
import subprocess

from setuptools import setup
from pkg_resources import get_build_platform
from wheel.bdist_wheel import bdist_wheel as bdist_wheel_

HERE = os.path.abspath(os.path.dirname(__file__))

NAME = "spirit"
PACKAGES = ["spirit", "spirit.parameters", "spirit_cli_tools"]
META_PATH = os.path.join("spirit", "__init__.py")
KEYWORDS = ["Spirit", "Spin Dynamics"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: C",
    "Programming Language :: C++",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
INSTALL_REQUIRES = ["numpy"]

###############################################################################


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file. Assume UTF-8 encoding, but replace Windows CRLF with Unix LF.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read().replace("\r\n", "\n")


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


class bdist_wheel(bdist_wheel_):
    def finalize_options(self):
        super().finalize_options()
        platform = os.environ.get("SPIRIT_PLATFORM_OVERRIDE", "")
        self.plat_name = platform if platform else get_build_platform()
        self.plat_name_supplied = True


def get_git_commit_datetime():
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8"
        ).strip()
        commit_datetime = subprocess.check_output(
            ["git", "show", "--quiet", "--format=%ci", commit_hash],
            encoding="utf-8",
        ).strip()
        commit_datetime = " ".join(commit_datetime.split()[:-1])
        datetime_object = datetime.datetime.strptime(
            commit_datetime, "%Y-%m-%d %H:%M:%S"
        )
        return "{:%Y%m%d%H%M%S}".format(datetime_object)
    except subprocess.CalledProcessError as cpe:
        print(cpe.output)
        return None


def make_version():
    # If the environment variable SPIRIT_ADD_VERSION_SUFFIX is defined,
    # it is appended to the package version number.
    bare_version = find_meta("version")
    add_version_suffix = os.environ.get("SPIRIT_ADD_VERSION_SUFFIX", "")
    if add_version_suffix.lower() in ("yes", "true", "t", "1"):
        timepoint_string = get_git_commit_datetime()
        if timepoint_string is None:
            timepoint_string = "{:%Y%m%d%H%M}".format(datetime.datetime.now())
        return f"{bare_version}.dev{timepoint_string}"
    else:
        return bare_version


if __name__ == "__main__":
    setup(
        name=NAME,
        python_requires=">=3.11",
        description=find_meta("description"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        license=find_meta("license"),
        url=find_meta("uri"),
        version=make_version(),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        keywords=KEYWORDS,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require={
            "dev": ["jinja2", "tree-sitter", "tree-sitter-cpp"],
        },
        entry_points={
            "console_scripts": [
                "spirit-mkinteraction = spirit_cli_tools.mkinteraction:cli [dev]",
                "spirit-cfgconvert = spirit_cli_tools.cfgconvert:cli",
            ]
        },
        package_data={
            "spirit": ["libSpirit.dylib", "libSpirit.so", "Spirit.dll"],
            "spirit_cli_tools": ["*.j2"],
        },
        cmdclass={"bdist_wheel": bdist_wheel},
    )
