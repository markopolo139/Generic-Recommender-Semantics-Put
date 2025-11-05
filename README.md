# Generic Recommender System

- Mateusz Bernart 156072
- Patryk Janiak 156053
- Marek Seget 156042

## Notebooks

To run code inside of the notebooks we recommend you to create a new Python's virtual environment. We tested the code with Python version `3.10.18`. Here is an example how to setup such a virtual environment with Python already installed and avaiable system-wide, to do so we will be using `uv`.

```console
uv venv env --python 3.10
source env/bin/activate
uv pip install -r notebooks/requirements.txt
```

After running this code you will be able to download the datasets using the `dvc` command.

```console
dvc pull
```

Now you are ready to run the code in the notebooks!
