# custom bash script to build the docs and put them in a place that is
# visible to github pages, which unfortunately cannot see into the
# build folder.
from multiprocessing.sharedctypes import Value
import os
import shutil
import sys

if __name__=='__main__':

    if 'notest' not in sys.argv:
        print("Verifying example scripts")

        example_path = os.path.join("source", "examples")
        os.chdir(example_path)

        for file in os.listdir("."):
            if file.startswith('example_') and file.endswith('.py'):
                print("Running " + file)
                out = os.system('python '+file)
                if out!=0:
                    raise ValueError("Example "+ file+ " is broken")
        print("All example scripts executed without errors.")
        os.chdir("..")
        os.chdir("..")

    def render_example_code_in_readme( path_to_raw_readme, example_code_path, pattern):
        """Handle code inclusion from file in github readme. Any line in the raw .rst that
        starts with ``pattern`` will be replaced by a github friendly ".. code::" block
        inserting all code lines from ``example_code_path`` that lies between two patterns
        in the ``example_code_path`.

        (The reason for this function is that github does not support code inclusion in the readme.)
        """
        with open(path_to_raw_readme, "r") as f:
            data = f.readlines()
        new_data = []
        for i, line in enumerate(data):
            if line.strip().startswith(pattern):
                example = str(line.strip())
                new_data.append("   .. code:: python")
                new_data.append("\n\n")
                with open(example_code_path, "r") as f:
                    read_lines = False
                    for code_line in f.readlines():

                        if code_line.strip().startswith(pattern) and not code_line.strip().startswith(example):
                            read_lines = False
                        if read_lines:
                            if len(code_line.strip()) > 0:
                                new_data.append("      " + code_line)
                            else:
                                new_data.append("\n")

                        if code_line.strip().startswith(example):
                            read_lines = True
            else:
                new_data.append(line)
        with open(os.path.join("..", "README.rst"), 'w') as f:
            f.writelines(new_data)

    def clean_readme_code( example_code_path, pattern):
        """Produce a .py file from all the code present in the README.
        """
        with open(example_code_path, "r") as f:
            data = f.readlines()
        new_data = []
        isreading=False
        for i, line in enumerate(data):
            if line.strip().startswith(pattern) and not isreading:
                isreading=True
            elif line.strip().startswith(pattern) and isreading:
                isreading=False
                new_data.append('\n')
            elif isreading:
                new_data.append(line)
        with open(os.path.join("source", "examples", "readme_tutorial.py"), 'w') as f:
            f.writelines(new_data)

    clean_readme_code( os.path.join("source", "examples", "example_readme.py"), 
                        pattern="##example:")

    render_example_code_in_readme(os.path.join("source", "raw_README.rst"),
                                os.path.join("source", "examples", "example_readme.py"),
                                pattern="##example:")


    out = os.system("make html")
    if out != 0:
        print("")
        print("")
        raise ValueError("Failed to build docs")
    html = os.path.join("build", "html")
    for file in os.listdir(html):
        if not file.startswith('_'):
            shutil.copy2(os.path.join(html, file), ".")

    for file in os.listdir(os.path.join(html, '_static')):
        if file.endswith('.png'):
            shutil.copy2(
                os.path.join(
                    os.path.join(
                        html,
                        '_static'),
                    file),
                "_static")
    print("Copied all docs to path visible by github pages")
