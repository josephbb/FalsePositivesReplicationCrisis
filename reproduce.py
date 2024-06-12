import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import time
import os

def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    t0 = time.time()

    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
    except Exception as e:
        print(f"Error executing the notebook '{notebook_path}': {e}")

    t1 = time.time()
    elapsed = t1 - t0
    print(f"Execution of {notebook_path} took {elapsed:.2f} seconds")
    return nb, elapsed

def save_notebook_as_html(nb, notebook_path):
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True

    (body, resources) = html_exporter.from_notebook_node(nb)


    print('Creating output directories')
    for directory in ["./output/", "./output/notebooks/"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    html_path =  "./output/notebooks/" + notebook_path.replace(".ipynb", ".html")
    with open(html_path, 'w') as f:
        f.write(body)
    print(f"Notebook saved as {html_path}")

def main():
    notebooks = ["./RepEfforts.ipynb",
                 "./FalsePositiveAnalysis.ipynb",
                 "./Theory.ipynb"]

    total_time = 0
    for nb in notebooks:
        print("------------------------")
        print("Running:" + nb)
        notebook, elapsed = run_notebook(nb)
        save_notebook_as_html(notebook, nb)
        total_time += elapsed

    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()