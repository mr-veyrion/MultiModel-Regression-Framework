# MultiModel-Regression-Framework/src/main.py

import argparse
import os
import sys
import pandas as pd

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from src.preprocessing import load_and_preprocess_data_v2
from src.training import train_and_evaluate_stacking_v2
import src.config as config

def run_pipeline(args):
    try:
        print("Loading and preprocessing data (v2)...")
        
        # args.data_dir is relative to the CWD when the script is called.
        # If you `cd src` and run `python main.py`, CWD is `.../src/`.
        # If args.data_dir is '../data', it correctly becomes `.../data/`.
        data_dir_to_use = os.path.abspath(args.data_dir) 

        train_features_path = os.path.join(data_dir_to_use, args.train_features)
        train_target_path = os.path.join(data_dir_to_use, args.train_target)
        test_features_path = os.path.join(data_dir_to_use, args.test_features)
        test_target_path = os.path.join(data_dir_to_use, args.test_target)

        print(f"CWD: {os.getcwd()}") # Debug: Print current working directory
        print(f"Resolved data_dir_to_use: {data_dir_to_use}") # Debug
        print(f"Attempting to load train features from: {train_features_path}")

        data_dict = load_and_preprocess_data_v2(
            train_features_path, train_target_path,
            test_features_path, test_target_path
        )
        
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        feature_names = data_dict['feature_names']

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        if X_train.shape[1] == 0:
             print(f"Warning: No features available after preprocessing. X_train.shape: {X_train.shape}")
             print(f"Feature names list: {feature_names}")
             return 
        print(f"Number of features after preprocessing: {X_train.shape[1]}")

        print("\nStarting model training, optimization, and evaluation (v2)...")
        model, metrics, test_predictions, oof_predictions = train_and_evaluate_stacking_v2(
            X_train, y_train, X_test, y_test, feature_names,
            n_trials_gbm=args.n_trials_gbm,
            n_trials_rf_et=args.n_trials_rf_et,
            n_trials_nn=args.n_trials_nn
        )
        print("\n--- Final Model Performance Metrics ---")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.5f}")
        
        if args.output_dir:
            output_dir_to_use = os.path.abspath(args.output_dir)
            os.makedirs(output_dir_to_use, exist_ok=True)
            
            oof_df = pd.DataFrame({'target': y_train, 'oof_pred': oof_predictions})
            oof_df.to_csv(os.path.join(output_dir_to_use, 'oof_predictions.csv'), index=False)
            
            test_pred_df = pd.DataFrame({config.TARGET_COLUMN_NAME: y_test, 'test_pred': test_predictions})
            test_pred_df.to_csv(os.path.join(output_dir_to_use, 'test_predictions.csv'), index=False)
            
            print(f"\nOOF and Test predictions saved to {output_dir_to_use}")

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please check paths. Details: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An error occurred in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    default_data_dir_relative_to_src = '../data'
    default_output_dir_relative_to_src = '../outputs'

    parser = argparse.ArgumentParser(description="Multi-Model Regression Training Framework")
    
    parser.add_argument('--data_dir', type=str, 
                        default=default_data_dir_relative_to_src, 
                        help=f'Directory containing data files (default: {default_data_dir_relative_to_src} relative to src CWD)')
    parser.add_argument('--train_features', type=str, 
                        default=config.DEFAULT_TRAIN_FEATURES_FILE, 
                        help=f'Filename for training features (default from config: {config.DEFAULT_TRAIN_FEATURES_FILE})')
    parser.add_argument('--train_target', type=str, 
                        default=config.DEFAULT_TRAIN_TARGET_FILE, 
                        help=f'Filename for training target (default from config: {config.DEFAULT_TRAIN_TARGET_FILE})')
    parser.add_argument('--test_features', type=str, 
                        default=config.DEFAULT_TEST_FEATURES_FILE, 
                        help=f'Filename for test features (default from config: {config.DEFAULT_TEST_FEATURES_FILE})')
    parser.add_argument('--test_target', type=str, 
                        default=config.DEFAULT_TEST_TARGET_FILE, 
                        help=f'Filename for test target (default from config: {config.DEFAULT_TEST_TARGET_FILE})')
    
    parser.add_argument('--n_trials_gbm', type=int, 
                        default=config.DEFAULT_N_TRIALS_GBM, 
                        help=f'Number of Optuna trials for GBM models (default from config: {config.DEFAULT_N_TRIALS_GBM})')
    parser.add_argument('--n_trials_rf_et', type=int, 
                        default=config.DEFAULT_N_TRIALS_RF_ET, 
                        help=f'Number of Optuna trials for RF/ET models (default from config: {config.DEFAULT_N_TRIALS_RF_ET})')
    parser.add_argument('--n_trials_nn', type=int, 
                        default=config.DEFAULT_N_TRIALS_NN, 
                        help=f'Number of Optuna trials for Neural Network (default from config: {config.DEFAULT_N_TRIALS_NN})')
    
    parser.add_argument('--output_dir', type=str, 
                        default=default_output_dir_relative_to_src, 
                        help=f'Directory to save prediction outputs (default: {default_output_dir_relative_to_src} relative to src CWD)')

    args = parser.parse_args()
    run_pipeline(args)
