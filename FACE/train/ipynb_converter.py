import nbformat
from nbconvert import PythonExporter


def convert_ipynb_to_py(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(source)

    print('done')


# ipynb_file_name = 'Train-Power.ipynb'
ipynb_file_name = 'Train-BJAQ.ipynb'
py_file_name = 'Train-BJAQ.py'
convert_ipynb_to_py(ipynb_file_name, py_file_name)
