"""Parse python files and look for inline sphinx comments that represents examples.

I.e comments that starts with ".. code-block:: python" are parsed and the python code is returned
as a strings.

We then write example code to a temp file module and import to check for any obvious errors.

This is essentially a simplified version of the doctest native module in python. It makes me feel
more comfortable to have it re-implemented by my own hand as doctesting inherently will make
exec() calls allowing for arbitrary code execution, should someone put malicious code in the
docstrings....

Axel Henningsson, 2026-01-02
"""

import os
import re
import textwrap
import unittest
from pathlib import Path


def collect_sphinx_examples(directory):
    """Recursively parse all python files in a directory for docstrings.

    I.e comments that starts with ".. code-block:: python" are parsed and
    the python code is returned as a strings. Note that the doc examples
    must end with 2 blank lines to be parsed correctly. This is the trigger
    for - end of example code -.

    Example:

        .. code-block:: python
        << blank line >>
            import numpy as np
            a = np.random.rand(10)
            a += 1
            plt.figure()
            plt.plot(a)
            plt.show()
        << blank line >>
        << blank line >>

        In the above example the following code will be executed:

            import numpy as np
            a = np.random.rand(10)
            a += 1
            plt.figure()
            plt.plot(a)
            # plt.show()

        Note that the plt.show() command is replaced with # plt.show() to avoid actually showing the figure.

    Args:
        directory (:obj: `str`): path to directory to traverse and parse.

    Returns:
        (:obj:`tuple`): A tuple containing two lists:
            - (:obj:`list` of `str`): The python files that contain doc-examples.
            - (:obj:`list` of `str`): The python code snippets extracted from the doc-examples.
    """
    # Pattern to capture entire Sphinx docstrings (triple-quoted strings)
    sphinx_docstring_pattern = re.compile(r"\"\"\"(.*?)\"\"\"", re.DOTALL)
    code_snippets = []
    python_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Find all Sphinx-style docstrings
                    for docstring in sphinx_docstring_pattern.findall(content):
                        reading = False
                        doc_lines = docstring.split("\n")
                        for i, line in enumerate(doc_lines):
                            # extract example code
                            if ".. code-block:: python" in line:
                                reading = True
                                example = ""
                                blank_line_counter = 0
                            elif reading:
                                example += line + "\n"
                                if line.strip() == "":
                                    blank_line_counter += 1
                                else:
                                    blank_line_counter = 0
                                if blank_line_counter == 2:
                                    reading = False
                                    example = textwrap.dedent(example)
                                    code_snippets.append(example)
                                    python_files.append(file)

    return python_files, code_snippets


def build_test_suite(test_file, modname):
    """Build a test suite for a given test file and module name.

    This function is used to dynamically build a test suite for a given test file and module name.
    The test suite will contain a test case for each doc-example in the module.
    The test case will execute the doc-example and check for obvious errors, i.e, if the code
    runs without raising an exception it auto-passes.

    Doc-examples are identified by the pattern ".. code-block:: python" in the docstring of the function
    and must terminate with exactly 2 blank lines. (See collect_sphinx_examples)

    Args:
        test_file (:obj:`str`): The path to the test file that is calling this function to dynamically build the test suite.
        modname (:obj:`str`): The name of the module that is to be tested. All doc-examples of the correct pattern in this
            module will be executed.

    Returns:
        (:obj:`unittest.TestSuite`): A test suite for the given test file and module name.
    """
    print(test_file)
    root = Path(__file__).resolve().parents[1]
    module_path = os.path.join(root, "darling", modname)
    cwd = Path(test_file).resolve().parent
    python_files, code_snippets = collect_sphinx_examples(module_path)
    suite = unittest.TestSuite()
    for pyfile, snippet in zip(python_files, code_snippets):
        suite.addTest(
            unittest.FunctionTestCase(
                lambda pyfile=pyfile, snippet=snippet, cwd=cwd: run_example(
                    pyfile, snippet, cwd
                ),
                description=str(pyfile),
            )
        )
    return suite


def run_example(pyfile, snippet, cwd):
    """execute strings of python code and check for obvious errors.

    Here the plt.show() command is replaced with # plt.show() to avoid actually showing the figure.

    Args:
        pyfile (:obj:`str`): The path to the python file that contains the doc-example.
        snippet (:obj:`str`): The python code-snippet extracted from the doc-example.
        cwd (:obj:`str`): The directury to execute the code in.

    Raises:
        (:obj:`ValueError`): If the example code-snippet fails to execute.
    """
    errmsg = ""
    snippet = snippet.replace("plt.show()", "# plt.show()")
    try:
        _exec(snippet, cwd=cwd)
    except Exception as e:
        msg = f"Sphinx docs example in {pyfile} failed to execute: \n>>\n{snippet}>> \n"
        errmsg += msg + "\n" + str(e) + "\n"
    if len(errmsg) > 0:
        raise ValueError(errmsg)


def _exec(code, cwd):
    # Simply execute the code in the given directory and return the result.
    old_cwd = os.getcwd()
    try:
        os.chdir(cwd)
        exec(code, {})
    finally:
        os.chdir(old_cwd)


class DocExamplesBoilerPlate:
    def __init__(self, modname, source_file):
        self.modname = modname
        self.source_file = source_file

    def run(self):
        # this will now dynamically build the test suite for the given module and source file...
        suite = build_test_suite(self.source_file, modname=self.modname)

        # and then run the test suite...
        result = unittest.TestResult()
        suite.run(result)

        # and then check for any errors or failures...
        if result.errors or result.failures:
            msgs = []
            for test, tb in result.errors + result.failures:
                msgs.append(f"{test}\n{tb}")
            raise AssertionError("\n\n".join(msgs))
