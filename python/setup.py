import os
import sys
import platform
from pathlib import Path

import numpy as np
import pybind11
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "1.0.8"


# Find the "cpp" folder depending on where this script is run from:
for search_path in ["./cpp/", "../cpp/", "../../cpp/"]:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), search_path))
    if os.path.exists(path):
        VOYAGER_HEADERS_PATH = path
        break
else:
    dir_contents = os.listdir(os.getcwd())
    raise OSError(
        "Unable to find the 'cpp' folder to build voyager. "
        f"Current working directory is: {os.getcwd()}, directory contains: "
        f"{', '.join([repr(x) for x in dir_contents[:5]])} and {len(dir_contents) - 5} more files."
    )


ext_modules = [
    Extension(
        "voyager",
        ["./bindings.cpp"],
        include_dirs=[pybind11.get_include(), np.get_include(), VOYAGER_HEADERS_PATH],
        libraries=[],
        language="c++",
        extra_objects=[],
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


DEBUG = int(os.environ.get("DEBUG", "0")) == 1
USE_ASAN = int(os.environ.get("USE_ASAN", "0")) == 1


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    compiler_flags = {
        "msvc": ["/EHsc", "/O2"],
        "unix": ["-O0" if DEBUG else "-O3"] + (["-g"] if DEBUG else []),
    }

    linker_flags = {"unix": [], "msvc": []}

    if sys.platform == "darwin":
        compiler_flags["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        linker_flags["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    if USE_ASAN:
        compiler_flags["unix"].append("-fsanitize=address")
        compiler_flags["unix"].append("-fno-omit-frame-pointer")
        linker_flags["unix"].append("-fsanitize=address")
        if platform.system() == "Linux":
            linker_flags["unix"].append("-shared-libasan")

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.compiler_flags.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++17")
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            opts.append("/std:c++17")

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.linker_flags.get(ct, []))

        build_ext.build_extensions(self)


project_root = Path(__file__).resolve().parent.parent
# Just for GitHub Actions and cibuildwheel:
cibuildwheel_project_root = Path("/home/runner/work/voyager/voyager")
if not project_root.exists() and cibuildwheel_project_root.exists():
    project_root = cibuildwheel_project_root
long_description = (project_root / "README.md").read_text("utf-8")

setup(
    name="voyager",
    version=__version__,
    description="Easy-to-use, fast, simple multi-platform approximate nearest-neighbor search.",
    author="Peter Sobot",
    url="https://github.com/spotify/voyager",
    long_description=long_description,
    ext_modules=ext_modules,
    install_requires=["numpy"],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
