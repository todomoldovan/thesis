# Measuring Commitment to Collective Grassroots Movements

This is the source code for the MSc thesis by [Theodora Moldovan]([https://www.linkedin.com/in/theodora-moldovan/]).

## Installation
First clone the repository:

```
git clone https://github.com/todomoldovan/thesis.git
```

Go to the cloned folder and create a new virtual environment. You can either create a new virtual environment then install the necessary dependencies with `pip` using the `requirements.txt` file:

```
pip install -r requirements.txt
```

Or create a new environment with the dependencies with `conda` or `mamba` using the `environment.yml` file:

```
mamba env create -f environment.yml
```
Then, install the virtual environment's kernel in Jupyter:

```
mamba activate ENVNAME
ipython kernel install --user --name=ENVNAME
mamba deactivate
```

You can now run `jupyter lab` with kernel `ENVNAME` (Kernel > Change Kernel > ENVNAME).

## Repository structure

```
├── data
│   ├── processed           <- Modified data
│   └── raw                 <- Original, immutable data
├── notebooks               <- Jupyter notebooks
├── plots                   <- Generated figures
├── scripts                 <- Scripts to execute
├── .gitignore              <- Files and folders ignored by git
├── .pre-commit-config.yaml <- Pre-commit hooks used
├── CITATION.cff            <- Citation file (template)
├── README.md
├── TEMPLATE.md             <- Explanation for the template, delete it after use
├── environment.yml         <- Environment file to set up the environment using conda/mamba
└── requirements.txt        <- Requirements file to set up the environment using pip
```


