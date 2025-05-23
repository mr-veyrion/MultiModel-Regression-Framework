
import argparse
import os
import sys
import pandas as pd # For saving predictions, if uncommented

# Ensure the 'src' directory is in sys.path if running from project root
# This allows for `from preprocessing import ...` even if main.py is moved
# or run via `python src/main.py` from the project root.
# current_file_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_file_dir) # Assuming src is one level down from project root
# if project_root not in sys.path:
#    sys.path.insert(0, project_root)
# Note: If main.py is in 'src' and you run `python main.py` from within 'src',
# relative imports `from .preprocessing import ...` are preferred.
# If you run `python src/main.py` from the project root, then Python handles 'src' as a package.

from preprocessing import load_and_preprocess_data_v2
from training import train_and_evaluate_stacking_v2
import config # Import the config module

def run_pipeline(args):
    try:
        print("Loading and preprocessing data (v2)...")
        
        # Use data_dir from args. If it's the default from config, it's already an absolute path.
        # If user provides a relative path via CLI, os.path.join will handle it from CWD.
        data_dir_to_use = args.data_dir
        
        train_features_path = os.path.join(data_dir_to_use, args.train_features)
        train_target_path = os.path.join(data_dir_to_use, args.train_target)
        test_features_path = os.path.join(data_dir_to_use, args.test_features)
        test_target_path = os.path.join(data_dir_to_use, args.test_target)

        print(f"Attempting to load train features from: {train_features_path}") # Debug print

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
            n_trials_gbm=args.n_trials_gbm, # argparse defaults will use config values
            n_trials_rf_et=args.n_trials_rf_et,
            n_trials_nn=args.n_trials_nn
        )
        print("\n--- Final Model Performance Metrics ---")
        for metric_name, value in metrics.items(): # Renamed metric to metric_name to avoid conflict
            print(f"{metric_name}: {value:.5f}")
        
        # Optional: Save predictions to specified output directory
        if args.output_dir:
            # If output_dir is relative, it will be created relative to where the script is run.
            # For consistency, you might want to make it relative to project root like data_dir.
            output_dir_to_use = args.output_dir 
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
    parser = argparse.ArgumentParser(description="Multi-Model Regression Training Framework")
    
    parser.add_argument('--data_dir', type=str, 
                        default=config.DEFAULT_DATA_DIR, 
                        help=f'Directory containing data files (default from config: {config.DEFAULT_DATA_DIR})')
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
                        default=config.DEFAULT_OUTPUT_DIR, # Use default from config
                        help=f'Directory to save prediction outputs (default from config: {config.DEFAULT_OUTPUT_DIR})')

    args = parser.parse_args()
    run_pipeline(args)
