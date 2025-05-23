
# Multi-Model Regression Framework 🚀

A Python framework for training and ensembling multiple regression models (XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Neural Network) using Optuna for hyperparameter tuning and `StackingRegressor` for ensembling. Designed for adaptability and achieving high predictive accuracy.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Running the Pipeline](#running-the-pipeline)
  - [Command-Line Arguments](#command-line-arguments)
- [Configuration](#configuration)
- [Modular Components](#modular-components)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Comprehensive Preprocessing:** Includes imputation, scaling (StandardScaler), and distribution normalization (PowerTransformer).
- **Advanced Feature Engineering:**
  - Interaction features between variable types.
  - Aggregate features (mean, standard deviation).
  - PCA for dimensionality reduction.
  - Polynomial interactions.
- **Multiple Base Regression Models:**
  - XGBoost, LightGBM, CatBoost
  - Random Forest, Extra Trees
  - Feedforward Neural Network (Keras)
- **Hyperparameter Optimization:** Optuna (TPESampler) for tuning.
- **Stacking Ensemble:** `StackingRegressor` using LightGBM as default meta-learner.
- **Evaluation:** Out-of-fold (OOF) RMSE, MAE, R² on test set.
- **Configurable:** Easily modifiable `config.py` or via CLI.
- **Modular Codebase:** Cleanly organized for extensibility.

## Requirements

- Python 3.8+
- Required Libraries: scikit-learn, NumPy, Pandas, XGBoost, LightGBM, CatBoost, Optuna, TensorFlow
- See `requirements.txt` for all dependencies.

## Installation

```bash
git clone https://github.com/mr-veyrion/MultiModel-Regression-Framework.git

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
````

## Usage

### Data Preparation

Ensure the following CSV files are placed in the directory defined in `config.py` (`data/` by default):

* `final_train_features.csv`
* `final_train_target.csv`
* `final_test_features.csv`
* `final_test_target.csv`

The target files must contain a single column named `target`. Adjust names in `config.py` or via CLI if needed.

### Running the Pipeline

From the project root:

```bash
cd src
python main.py
```

To override defaults:

```bash
python main.py --data_dir ../data --n_trials_gbm 25 --n_trials_nn 15
```

### Command-Line Arguments

* `--data_dir`: Path to data files
* `--train_features`, `--train_target`, `--test_features`, `--test_target`: Override file names
* `--n_trials_gbm`, `--n_trials_rf_et`, `--n_trials_nn`: Optuna trial count per model group
* `--output_dir`: Directory to save predictions

Run `python main.py --help` for all options.

## Configuration

Centralised in `config.py`. Editable options include:

* Data paths & file names
* Number of Optuna trials
* Random seeds
* Output directory
* Meta-learner & stacking parameters
* Neural network architecture

Command-line options override config defaults.

## Modular Components

* `config.py`: All configs
* `feature_engineering.py`: Feature creation
* `model_definitions.py`: Model architectures and wrappers
* `optimization.py`: Optuna tuning objectives
* `preprocessing.py`: Imputation, scaling, transformations
* `training.py`: Training, evaluation, stacking
* `main.py`: CLI interface and pipeline runner

## Future Enhancements

* Add classification support
* Advanced feature selection (RFE, Boruta)
* Meta-learner hyperparameter tuning
* Model persistence and loading
* MLflow/Weights & Biases integration
* Selective base model inclusion
* Automated reports and visualisation


## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

Thanks to the amazing open-source tools that power this framework:

* [Scikit-learn](https://scikit-learn.org/)
* [Optuna](https://optuna.org/)
* [XGBoost](https://xgboost.ai/)
* [LightGBM](https://lightgbm.readthedocs.io/)
* [CatBoost](https://catboost.ai/)
* [TensorFlow](https://www.tensorflow.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)

```
