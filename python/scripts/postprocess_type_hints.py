#! /usr/bin/env python
#
# Copyright 2022-2023 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is intended to postprocess type hint (.pyi) files produced by pybind11-stubgen.

A number of changes must be made to produce valid type hint files out-of-the-box.
Think of this as a more intelligent (or less intelligent) application of `git patch`.
"""

import os
import io
import re
import difflib
from pathlib import Path
from collections import defaultdict
import black
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OMIT_FILES = []

# Any lines containing any of these substrings will be omitted from the voyager stubs:
OMIT_LINES_CONTAINING = []

REPLACEMENTS = [
    # object is a superclass of `str`, which would make these declarations ambiguous:
    ("file_like: object", "file_like: typing.BinaryIO"),
    # "r" is the default file open/reading mode:
    ("mode: str = 'r'", r'mode: Literal["r"] = "r"'),
    # ... but when using "w", "w" needs to be specified explicitly (no default):
    ("mode: str = 'w'", r'mode: Literal["w"]'),
    # ndarrays need to be corrected as well:
    (r"numpy\.ndarray\[(.*?)\]", r"numpy.ndarray[typing.Any, numpy.dtype[\1]]"),
    # None of our enums are properly detected by pybind11-stubgen:
    (
        # For Python 3.6 compatibility:
        r"import typing",
        "\n".join(
            ["import typing", "from typing_extensions import Literal", "from enum import Enum"]
        ),
    ),
    # Remove type hints in docstrings, added unnecessarily by pybind11-stubgen
    (r".*?:type:.*$", ""),
    # MyPy chokes on classes that contain both __new__ and __init__.
    # Remove all bare, arg-free inits:
    (r"def __init__\(self\) -> None: ...", ""),
    (
        "import typing",
        """
import typing

original_overload = typing.overload
__OVERLOADED_DOCSTRINGS = {}

def patch_overload(func):
    original_overload(func)
    if func.__doc__:
        __OVERLOADED_DOCSTRINGS[func.__qualname__] = func.__doc__
    else:
        func.__doc__ = __OVERLOADED_DOCSTRINGS.get(func.__qualname__)
    if func.__doc__:
        # Work around the fact that pybind11-stubgen generates
        # duplicate docstrings sometimes, once for each overload:
        while func.__doc__[len(func.__doc__) // 2:].strip() \\
                == func.__doc__[:-len(func.__doc__) // 2].strip():
            func.__doc__ = func.__doc__[len(func.__doc__) // 2:].strip()
    return func

typing.overload = patch_overload
    """,
    ),
]

REMOVE_INDENTED_BLOCKS_STARTING_WITH = []


# .pyi files usually don't care about dependent types being present in order in the file;
# but if we want to use Sphinx, our Python files need to be both lexically and semantically
# valid as regular Python files. So...
INDENTED_BLOCKS_TO_MOVE_TO_END = []

# Similar to above; move our enum types early in the file (after the end of __all__ = [...])
INDENTED_BLOCKS_TO_MOVE_TO_START = [
    "class Space(Enum)",
    "class StorageDataType(Enum)",
]

LINES_TO_IGNORE_FOR_MATCH = {"from __future__ import annotations"}


def stub_files_match(a: str, b: str) -> bool:
    a = "".join(
        [x for x in a.split("\n") if x.strip() and x.strip() not in LINES_TO_IGNORE_FOR_MATCH]
    )
    b = "".join(
        [x for x in b.split("\n") if x.strip() and x.strip() not in LINES_TO_IGNORE_FOR_MATCH]
    )
    return a == b


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Post-process type hint files produced by pybind11-stubgen for Voyager."
    )
    parser.add_argument(
        "source_directory",
        default=os.path.join(REPO_ROOT, "pybind11-stubgen-output"),
    )
    parser.add_argument(
        "target_directory",
        default=os.path.join(REPO_ROOT, "voyager"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Return a non-zero exit code if files on disk don't match what this script would"
            " generate."
        ),
    )
    args = parser.parse_args(args)

    output_file_to_source_files = defaultdict(list)
    for source_path in Path(args.source_directory).rglob("*.pyi"):
        output_file_to_source_files[str(source_path)].append(str(source_path))

    for output_file_name, source_files in output_file_to_source_files.items():
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

        print(f"Writing stub file {output_file_name}...")

        file_contents = io.StringIO()
        start_of_file_contents = io.StringIO()
        end_of_file_contents = io.StringIO()
        for source_file in source_files:
            module_name = output_file_name.replace("__init__.pyi", "").replace("/", ".").rstrip(".")
            with open(source_file) as f:
                in_excluded_indented_block = False
                in_moved_to_start_indented_block = False
                in_moved_to_end_indented_block = False
                for line in f:
                    if all(x not in line for x in OMIT_LINES_CONTAINING):
                        if any(line.startswith(x) for x in REMOVE_INDENTED_BLOCKS_STARTING_WITH):
                            in_excluded_indented_block = True
                            continue
                        elif any(line.startswith(x) for x in INDENTED_BLOCKS_TO_MOVE_TO_END):
                            in_moved_to_end_indented_block = True
                        elif any(line.startswith(x) for x in INDENTED_BLOCKS_TO_MOVE_TO_START):
                            in_moved_to_start_indented_block = True
                        elif line.strip() and not line.startswith(" "):
                            in_excluded_indented_block = False
                            in_moved_to_end_indented_block = False

                        if in_excluded_indented_block:
                            continue

                        for _tuple in REPLACEMENTS:
                            if len(_tuple) == 2:
                                find, replace = _tuple
                                only_in_module = None
                            else:
                                find, replace, only_in_module = _tuple
                            if only_in_module and only_in_module != module_name:
                                continue
                            results = re.findall(find, line)
                            if results:
                                line = re.sub(find, replace, line)

                        if in_moved_to_start_indented_block:
                            start_of_file_contents.write(line)
                        elif in_moved_to_end_indented_block:
                            end_of_file_contents.write(line)
                        else:
                            file_contents.write(line)
                print(f"\tRead {f.tell():,} bytes of stubs from {source_file}.")

        # Append end-of-file contents at the end:
        file_contents.write("\n")
        file_contents.write(end_of_file_contents.getvalue())

        # Find the end of the __all__ block (first line with only "]"):
        file_contents_string = file_contents.getvalue()
        start_of_file_marker = "\n]\n"
        insert_index = file_contents_string.find(start_of_file_marker)
        if insert_index < 0 and len(start_of_file_contents.getvalue()):
            raise NotImplementedError("Couldn't find end of __all__ block to move code blocks to!")
        insert_index += len(start_of_file_marker)

        file_contents_string = (
            file_contents_string[:insert_index]
            + "\n"
            + start_of_file_contents.getvalue()
            + "\n"
            + file_contents_string[insert_index:]
        )

        # Run black:
        try:
            output = black.format_file_contents(
                file_contents_string,
                fast=False,
                mode=black.FileMode(is_pyi=True, line_length=100),
            )
        except black.report.NothingChanged:
            output = file_contents_string

        if args.check:
            with open(output_file_name, "r") as f:
                existing = f.read()
                if not stub_files_match(existing, output):
                    error = f"File that would be generated ({output_file_name}) "
                    error += "does not match existing file!\n"
                    error += f"Existing file had {len(existing):,} bytes, "
                    error += f"expected {len(output):,} bytes.\nDiff was:\n"
                    diff = difflib.context_diff(existing.split("\n"), output.split("\n"))
                    error += "\n".join([x.strip() for x in diff])
                    raise ValueError(error)
        else:
            with open(output_file_name, "w") as o:
                o.write(output)
                print(f"\tWrote {o.tell():,} bytes of stubs to {output_file_name}.")

    print("Done!")


if __name__ == "__main__":
    main()
