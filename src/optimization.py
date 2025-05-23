
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf # For tf.keras.backend.clear_session()

# Import Keras wrapper from model_definitions
from .model_definitions import KerasRegressorWrapperV2 # Relative import

# DEBUGGING VERSION of Optuna functions
def optimize_xgb_v2(X, y, n_trials=5): 
    def objective(trial):
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'lambda': trial.suggest_float('lambda', 1e-3, 1.0, log=True), 
            'alpha': trial.suggest_float('alpha', 1e-3, 1.0, log=True), 
            'subsample': trial.suggest_float('subsample', 0.7, 1.0), 
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 
            'n_estimators': trial.suggest_int('n_estimators', 100, 300), 
            'max_depth': trial.suggest_int('max_depth', 3, 5), 
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5), 
            'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True), 
            'early_stopping_rounds': 10, 'random_state': 42, 'n_jobs': -1 
        }
        model = xgb.XGBRegressor(**params)
        scores = []
        kf = KFold(n_splits=2, shuffle=True, random_state=trial.number) 
        for train_idx, val_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx], 
                      eval_set=[(X[val_idx], y[val_idx])], 
                      verbose=False)
            preds = model.predict(X[val_idx])
            scores.append(mean_squared_error(y[val_idx], preds, squared=False))
        current_score = np.mean(scores)
        return current_score

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True) 
    print(f"XGBoost Optuna finished. Best params: {study.best_params}")
    return study.best_params


def optimize_lgb_v2(X, y, n_trials=5): 
    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']), 
            'n_estimators': trial.suggest_int('n_estimators', 100, 300), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50), 
            'max_depth': trial.suggest_int('max_depth', -1, 7), 
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 20),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        }
        model = lgb.LGBMRegressor(**params)
        scores = []
        kf = KFold(n_splits=2, shuffle=True, random_state=trial.number + 1) 
        for train_idx, val_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], eval_metric='rmse',
                      callbacks=[lgb.early_stopping(10, verbose=False)]) 
            preds = model.predict(X[val_idx])
            scores.append(mean_squared_error(y[val_idx], preds, squared=False))
        current_score = np.mean(scores)
        return current_score
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=123))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    print(f"LGBM Optuna finished. Best params: {study.best_params}")
    return study.best_params

def optimize_ctb_v2(X, y, n_trials=5): 
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 5),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 128),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 1.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'od_type': 'Iter', 'od_wait': 10, 
            'random_state': 42, 'verbose': 0, 'loss_function': 'RMSE', 'eval_metric': 'RMSE'
        }
        model = ctb.CatBoostRegressor(**params)
        scores = []
        kf = KFold(n_splits=2, shuffle=True, random_state=trial.number + 2) 
        for train_idx, val_idx in kf.split(X,y):
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=0)
            preds = model.predict(X[val_idx])
            scores.append(mean_squared_error(y[val_idx], preds, squared=False))
        current_score = np.mean(scores)
        return current_score
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=234))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    print(f"CatBoost Optuna finished. Best params: {study.best_params}")
    return study.best_params

def optimize_rf_v2(X, y, n_trials=3): 
    def objective(trial):
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 150), 'max_depth': trial.suggest_int('max_depth', 3, 7), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']), 'bootstrap': True, 'random_state': 42, 'n_jobs': -1}
        model = RandomForestRegressor(**params)
        score = cross_val_score(model, X, y, cv=KFold(2, shuffle=True, random_state=trial.number+3), scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return -score 
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=345)); study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True); print(f"RF Optuna finished. Best params: {study.best_params}"); return study.best_params

def optimize_extratrees_v2(X, y, n_trials=3): 
    def objective(trial):
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 150), 'max_depth': trial.suggest_int('max_depth', 3, 7), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']), 'bootstrap': False, 'random_state': 42, 'n_jobs': -1}
        model = ExtraTreesRegressor(**params)
        score = cross_val_score(model, X, y, cv=KFold(2, shuffle=True, random_state=trial.number+4), scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return -score
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=456)); study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True); print(f"ET Optuna finished. Best params: {study.best_params}"); return study.best_params

def optimize_nn_v2(X, y, n_trials=3): 
    def objective(trial):
        units1, units2, units3 = trial.suggest_categorical('units1', [64, 128]), trial.suggest_categorical('units2', [32, 64]), trial.suggest_categorical('units3', [16, 32])
        activation_other = trial.suggest_categorical('activation_other', ['relu', 'elu'])
        use_selu_alpha = trial.suggest_categorical('use_selu_alpha', [False]) # For debugging, force False, Optuna returns this key
        act1, act2, act3 = (activation_other,)*3 # Simpler activation scheme for debugging
        epochs, batch_size = trial.suggest_int('epochs', 30, 50), trial.suggest_categorical('batch_size', [64, 128])
        lr, dropout_rate = trial.suggest_float('lr', 1e-4, 1e-3, log=True), trial.suggest_float('dropout_rate', 0.1, 0.3)
        optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam'])
        
        kf = KFold(n_splits=2, shuffle=True, random_state=trial.number + 5) 
        fold_scores = []
        for train_idx, val_idx in kf.split(X,y):
            nn_wrapper = KerasRegressorWrapperV2(
                input_shape=(X.shape[1],), epochs=epochs, batch_size=batch_size, lr=lr, 
                dropout_rate=dropout_rate, units1=units1, units2=units2, units3=units3, 
                use_alpha_dropout=use_selu_alpha, # Pass the value determined by Optuna
                activation1=act1, activation2=act2, activation3=act3, 
                optimizer_name=optimizer_name
            )
            nn_wrapper.fit(X[train_idx], y[train_idx], validation_data=(X[val_idx], y[val_idx])) 
            y_pred = nn_wrapper.predict(X[val_idx])
            fold_scores.append(mean_squared_error(y[val_idx], y_pred, squared=False))
            tf.keras.backend.clear_session() # Important for Keras in loops
        return np.mean(fold_scores)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=567))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    print(f"NN Optuna finished. Best params: {study.best_params}")
    return study.best_params
