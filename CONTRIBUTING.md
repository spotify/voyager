# How to Contribute
We'd love to get patches from you!

#### Workflow
We follow the [GitHub Flow Workflow](https://guides.github.com/introduction/flow/):

1.  Fork the project 
1.  Check out the `master` branch 
1.  Create a feature branch
1.  Write code and tests for your change 
1.  From your branch, make a pull request against `https://github.com/spotify/voyager` 
1.  Work with repo maintainers to get your change reviewed 
1.  Wait for your change to be pulled into `https://github.com/spotify/voyager/master`
1.  Delete your feature branch

## Getting Started
### Prerequisites
To compile Voyager from scratch, the following packages will need to be installed:

- [Python 3.7](https://www.python.org/downloads/) or higher.
- A C++ compiler, e.g. `gcc`, `clang`, etc.

### Building Voyager
#### Building Python
There are some nuances to building the Voyager python code.  Please read on for more information.

For basic building, you should be able to simply run the following commands:
```shell
cd python
pip install -r python/dev-requirements.txt
pip install .
```

To compile a debug build of `voyager` that allows using a debugger (like gdb or lldb), use the following command to build the package locally and install a symbolic link for debugging:
```shell
cd python
DEBUG=1 python setup.py build develop
```

> If you're on macOS or Linux, you can try to compile a debug build _faster_ by using [Ccache](https://ccache.dev/):
> ## macOS
> ```shell
> brew install ccache
> rm -rf build && CC="ccache clang" CXX="ccache clang++" DEBUG=1 python -j8 -m pip install -e .
> ```
> ## Linux
> e.g.
> ```shell
> sudo yum install ccache  # or apt, if on a Debian
> 
> # If using GCC:
> rm -rf build && CC="ccache gcc" CXX="scripts/ccache_g++" DEBUG=1 python setup.py build -j8 develop
> 
> # ...or if using Clang:
> rm -rf build && CC="ccache clang" CXX="scripts/ccache_clang++" DEBUG=1 python setup.py build -j8 develop
> ```

#### Building Java
To build the Java library with `maven`, use the following commands:
```shell
cd java
mvn package
```

#### Building C++
To build the C++ library with `cmake`, use the following commands:
```shell
cd cpp
git submodule update --init --recursive
cmake .
make
```

## Testing
### Python Tests
We use `tox` for testing - running tests from end-to-end should be as simple as:

```
cd python
pip3 install tox
tox
```

## Benchmarking

We use `airspeed-velocity` for benchmarking - running all benchmarks to compare latest 
commit against previous should be as simple as:

```
pip3 install asv
asv continuous --sort name --no-only-changed HEAD HEAD~1
```

To compare current commit against `main` branch:

```
pip3 install asv
asv continuous --sort name --no-only-changed HEAD main
```

Please note that `airspeed-velocity` can only run benchmarks against a git commit, so if 
you have uncommited code that you want to run benchmarks for you need to commit it first.

### Java Tests
We provide java test execution as a maven test step.  Thus you can run the tests with:

```shell
cd java
mvn verify
```

### C++ Tests
To run the C++ tests, use the following commands:
```shell
cd cpp
git submodule update --init --recursive
cmake .
make test
./test/test
```

## Style
Use [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html) for C++ code, and `black` with defaults for Python code.

In order to check and run formatting within the python module (but not the c++ core module), you can use tox to facilitate this.
```bash
cd python
# Check formatting only (don't change files)
tox -e check-formatting
# Run formatter for python bindings and native python code
tox -e format
```

For C++ code within the core `cpp` module, you can use the following command to check formatting:
```bash
cd cpp
# Check formatting only
clang-format --verbose --dry-run -i src/*.h
# Run formatter
clang-format --verbose -i src/*.h
```

### Updating Documentation
We also welcome improvements to the project documentation or to the existing
docs. Please file an [issue](https://github.com/spotify/voyager/issues/new).

If you notice that the generated API documentation is out of date, feel free to run these commands in order to update the docs and make a PR with the changes.

#### Python
While `voyager` is mostly C++ code, it ships with `.pyi` files to allow for type hints in text editors and via MyPy. To update the Python type hint files, use the following commands:

```shell
cd python
python3 -m scripts.generate_type_stubs_and_docs
# Documentation will be dumped into ../docs/python/
```

#### Java
To update the javadocs for the java bindings, you can simply run:

```shell
cd java
mvn package
```

this will update the java documentation located in [docs/java/](https://github.com/spotify/voyager/tree/main/docs/java).

## Issues
When creating an issue please try to ahere to the following format:

    One line summary of the issue (less than 72 characters)

    ### Expected behaviour

    As concisely as possible, describe the expected behaviour.

    ### Actual behaviour

    As concisely as possible, describe the observed behaviour.

    ### Steps to reproduce the behaviour

    List all relevant steps to reproduce the observed behaviour.

## First Contributions
If you are a first time contributor to `voyager`,  familiarize yourself with the:
* [Code of Conduct](CODE_OF_CONDUCT.md)
* [GitHub Flow Workflow](https://guides.github.com/introduction/flow/)
<!-- * Issue and pull request style guides -->

When you're ready, navigate to [issues](https://github.com/spotify/voyager/issues/new). Some issues have been identified by community members as [good first issues](https://github.com/spotify/voyager/labels/good%20first%20issue). 

There is a lot to learn when making your first contribution. As you gain experience, you will be able to make contributions faster. You can submit an issue using the [question](https://github.com/spotify/voyager/labels/question) label if you encounter challenges.  

# License 
By contributing your code, you agree to license your contribution under the 
terms of the [LICENSE](https://github.com/spotify/voyager/blob/master/LICENSE).

# Code of Conduct
Read our [Code of Conduct](CODE_OF_CONDUCT.md) for the project.

# Troubleshooting
## Building the project
### `ModuleNotFoundError: No module named 'pybind11'`
Try updating your version of `pip`:
```shell
pip install --upgrade pip
```

### `Failed to establish a new connection: [Errno -2] Name or service not known'`
You may have networking issues. Check to make sure you do not have the `PIP_INDEX_URL` environment variable set (or that it points to a valid index).

### `fatal error: Python.h: No such file or directory`
Ensure you have the Python development packages installed.
You will need to find correct package for your operating system. (i.e.: `python-dev`, `python-devel`, etc.)

### `AttributeError: 'NoneType' object has no attribute 'group'`
- Ensure that you have Tox version 4 or greater installed
- _or_ set `ignore_basepython_conflict=true` in `tox.ini`
- _or_ install Tox using `pip` and not your system package manager
