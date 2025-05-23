
import os

# --- Project Root Directory ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Data File Configuration ---
# Default directory for data, relative to the project root.
# Used if --data_dir is not provided to main.py
DEFAULT_DATA_DIR_RELATIVE_TO_PROJECT_ROOT = "data"
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, DEFAULT_DATA_DIR_RELATIVE_TO_PROJECT_ROOT)

# Default filenames within the data directory
DEFAULT_TRAIN_FEATURES_FILE = "final_train_features.csv"
DEFAULT_TRAIN_TARGET_FILE = "final_train_target.csv"
DEFAULT_TEST_FEATURES_FILE = "final_test_features.csv"
DEFAULT_TEST_TARGET_FILE = "final_test_target.csv"

# Name of the target column in your target CSV files
TARGET_COLUMN_NAME = 'target'


# --- Optuna Hyperparameter Optimization Settings ---
# Default number of trials. Can be overridden by command-line arguments.
# Set these low for quick debugging, increase for thorough tuning.
DEFAULT_N_TRIALS_GBM = 2
DEFAULT_N_TRIALS_RF_ET = 2
DEFAULT_N_TRIALS_NN = 2

# Seeds for Optuna Samplers for reproducibility of optimization process
OPTUNA_SAMPLER_SEED_XGB = 42
OPTUNA_SAMPLER_SEED_LGB = 123
OPTUNA_SAMPLER_SEED_CTB = 234
OPTUNA_SAMPLER_SEED_RF = 345
OPTUNA_SAMPLER_SEED_ET = 456
OPTUNA_SAMPLER_SEED_NN = 567


# --- Model Training Seeds ---
# Global random seed for operations like train/test split, KFold, model initializations (where applicable)
GLOBAL_RANDOM_STATE = 42


# --- Output Configuration ---
# Default directory for saving outputs (predictions, models, etc.), relative to project root
DEFAULT_OUTPUT_DIR_RELATIVE_TO_PROJECT_ROOT = "outputs"
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, DEFAULT_OUTPUT_DIR_RELATIVE_TO_PROJECT_ROOT)


# --- Feature Engineering Specifics ---

PCA_N_COMPONENTS_SUBJECT_MAX = 3
PCA_N_COMPONENTS_BEHAVIOR_MAX = 2
POLYNOMIAL_DEGREE = 2
POLYNOMIAL_INTERACTION_ONLY = True


# --- Keras Neural Network Defaults (if not tuned by Optuna or for fixed parts) ---
# These are mostly for the KerasRegressorWrapperV2 defaults if Optuna doesn't override them.
# Optuna will suggest values for most of these during its search.
NN_DEFAULT_EPOCHS = 100
NN_DEFAULT_BATCH_SIZE = 64
NN_DEFAULT_LR = 1e-3
NN_DEFAULT_DROPOUT = 0.3
# Default architecture (Optuna will vary these)
NN_DEFAULT_UNITS1 = 128
NN_DEFAULT_UNITS2 = 64
NN_DEFAULT_UNITS3 = 32
NN_DEFAULT_ACTIVATION = 'relu'
NN_DEFAULT_OPTIMIZER = 'Adam'


# --- Stacking Ensemble Configuration ---
# KFold splits for training base models for meta-learner input
STACKING_CV_SPLITS_BASE_MODELS = 2 # Debug value, increase to 3 or 5
STACKING_CV_SHUFFLE_BASE_MODELS = True
STACKING_CV_RANDOM_STATE_BASE_MODELS = 111

# KFold splits for generating Out-Of-Fold (OOF) predictions for the entire stack
STACKING_OOF_CV_SPLITS = 3 # Debug value, increase to 5
STACKING_OOF_CV_SHUFFLE = True
STACKING_OOF_CV_RANDOM_STATE = 222

STACKING_PASSTHROUGH = True # Let meta-learner see original features

# Meta-learner fixed parameters (could also be tuned)
META_LEARNER_PARAMS = {
    'n_estimators': 100, # Debug value
    'learning_rate': 0.05,
    'num_leaves': 15, # Debug value
    'max_depth': 4,   # Debug value
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'random_state': GLOBAL_RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
    'objective': 'regression_l1'
}

# You can add more configuration variables as your project grows.
# For example, which models to include/exclude in the ensemble.
MODELS_TO_INCLUDE = ['xgb', 'lgb', 'ctb', 'rf', 'et', 'nn'] # All current models
