
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Relative imports for Optuna functions and Keras wrapper
from .optimization import (
    optimize_xgb_v2, optimize_lgb_v2, optimize_ctb_v2,
    optimize_rf_v2, optimize_extratrees_v2, optimize_nn_v2
)
from .model_definitions import KerasRegressorWrapperV2


def train_and_evaluate_stacking_v2(X_train, y_train, X_test, y_test, feature_names,
                                   n_trials_gbm=2, n_trials_rf_et=2, n_trials_nn=2): # Use defaults from argparse
    # Optuna trials from arguments
    N_TRIALS_GBM = n_trials_gbm
    N_TRIALS_RF_ET = n_trials_rf_et
    N_TRIALS_NN = n_trials_nn

    print("Optimizing XGBoost...")
    best_xgb_params = optimize_xgb_v2(X_train, y_train, n_trials=N_TRIALS_GBM)
    # print(f"Best XGBoost params: {best_xgb_params}") # Already printed in optimize func

    print("Optimizing LightGBM...")
    best_lgb_params = optimize_lgb_v2(X_train, y_train, n_trials=N_TRIALS_GBM)
    # print(f"Best LightGBM params: {best_lgb_params}")

    print("Optimizing CatBoost...")
    best_ctb_params = optimize_ctb_v2(X_train, y_train, n_trials=N_TRIALS_GBM)
    # print(f"Best CatBoost params: {best_ctb_params}")
    
    print("Optimizing RandomForest...")
    best_rf_params = optimize_rf_v2(X_train, y_train, n_trials=N_TRIALS_RF_ET)
    # print(f"Best RandomForest params: {best_rf_params}")

    print("Optimizing ExtraTrees...")
    best_et_params = optimize_extratrees_v2(X_train, y_train, n_trials=N_TRIALS_RF_ET)
    # print(f"Best ExtraTrees params: {best_et_params}")

    print("Optimizing Neural Network...")
    best_nn_params_from_optuna = optimize_nn_v2(X_train, y_train, n_trials=N_TRIALS_NN)
    # print(f"Best NN params from Optuna: {best_nn_params_from_optuna}")
    
    # Reconstruct actual NN parameters for the wrapper
    final_nn_params = best_nn_params_from_optuna.copy()
    # These keys are part of Optuna's trial suggestion, not direct KerasWrapperV2 __init__ args
    activation_other_val = final_nn_params.pop('activation_other', 'relu') # Default if somehow missing
    use_selu_alpha_val = final_nn_params.pop('use_selu_alpha', False)    # Default if somehow missing
    
    activation_selu_const = 'selu' # Matching definition in optimize_nn_v2
    if use_selu_alpha_val:
        final_nn_params['activation1'] = activation_selu_const
        final_nn_params['activation2'] = activation_selu_const
        final_nn_params['activation3'] = activation_selu_const
        final_nn_params['use_alpha_dropout'] = True # This should be passed to KerasWrapperV2
    else:
        final_nn_params['activation1'] = activation_other_val
        final_nn_params['activation2'] = activation_other_val
        final_nn_params['activation3'] = activation_other_val
        final_nn_params['use_alpha_dropout'] = False # This should be passed

    # Ensure 'use_alpha_dropout' is always in final_nn_params for the KerasWrapperV2 __init__
    # if it wasn't part of the tuned params (though it is in optimize_nn_v2)
    if 'use_alpha_dropout' not in final_nn_params:
         final_nn_params['use_alpha_dropout'] = use_selu_alpha_val


    estimators = [
        ('xgb', xgb.XGBRegressor(**{k: v for k, v in best_xgb_params.items()}, random_state=42)),
        ('lgb', lgb.LGBMRegressor(**{k: v for k, v in best_lgb_params.items()}, random_state=42)),
        ('ctb', ctb.CatBoostRegressor(**{k: v for k, v in best_ctb_params.items()}, random_state=42, verbose=0)),
        ('rf', RandomForestRegressor(**{k: v for k, v in best_rf_params.items()}, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(**{k: v for k, v in best_et_params.items()}, random_state=42, n_jobs=-1)),
        ('nn', KerasRegressorWrapperV2(input_shape=(X_train.shape[1],), **final_nn_params))
    ]
    # Simplified meta-learner for debugging
    meta_learner = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=15, max_depth=4, 
        subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1, 
        verbose=-1, objective='regression_l1' 
    )
    stacking_model = StackingRegressor(
        estimators=estimators, final_estimator=meta_learner,
        cv=KFold(n_splits=2, shuffle=True, random_state=111), # DEBUG: Reduced CV
        n_jobs=1, passthrough=True 
    )

    print("\nTraining Stacking Ensemble for OOF predictions...")
    oof_predictions = cross_val_predict(stacking_model, X_train, y_train, 
                                        cv=KFold(n_splits=3, shuffle=True, random_state=222), # DEBUG: Reduced CV
                                        n_jobs=1, method='predict') 

    print("Fitting final stacking model on full training data...")
    stacking_model.fit(X_train, y_train)
    final_test_pred = stacking_model.predict(X_test)

    metrics = {
        'oof_rmse': mean_squared_error(y_train, oof_predictions, squared=False),
        'test_rmse': mean_squared_error(y_test, final_test_pred, squared=False),
        'test_mae': mean_absolute_error(y_test, final_test_pred),
        'test_r2': r2_score(y_test, final_test_pred)
    }
    return stacking_model, metrics, final_test_pred, oof_predictions
