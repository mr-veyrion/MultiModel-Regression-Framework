
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from .feature_engineering import advanced_feature_engineering_v2, add_polynomial_features_v2 # Relative import

def load_and_preprocess_data_v2(train_features_path, train_target_path, test_features_path, test_target_path):
    try:
        train_features_raw = pd.read_csv(train_features_path)
        train_target = pd.read_csv(train_target_path)['target']
        test_features_raw = pd.read_csv(test_features_path)
        test_target = pd.read_csv(test_target_path)['target']
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        raise
    except KeyError as e:
        print(f"Error: 'target' column not found in one of the target CSVs: {e}")
        raise


    train_features_eng, pca_subject_model, pca_behavior_model = advanced_feature_engineering_v2(
        train_features_raw, is_train=True
    )
    test_features_eng = advanced_feature_engineering_v2(
        test_features_raw, is_train=False,
        pca_subject_model=pca_subject_model, pca_behavior_model=pca_behavior_model
    )

    common_cols = list(set(train_features_eng.columns) & set(test_features_eng.columns))
    if not common_cols:
        print("Error: No common columns found between engineered train and test features. Aborting.")
        # This case needs robust handling, perhaps by returning empty structures or raising error.
        # For now, let's assume it won't happen with current FE or be caught by X_train.shape[1] == 0 later
        return {'X_train': np.array([]), 'y_train': np.array([]), 'X_test': np.array([]), 'y_test': np.array([]), 
                'feature_names': [], 'scaler': None, 'power_transformer': None, 'imputer': None, 
                'pca_subject_model': None, 'pca_behavior_model': None}


    train_df_aligned = train_features_eng[common_cols]
    test_df_aligned = test_features_eng[common_cols]
    
    sorted_common_cols = sorted(common_cols)
    train_df_aligned = train_df_aligned[sorted_common_cols]
    test_df_aligned = test_df_aligned[sorted_common_cols]
    
    current_feature_names = train_df_aligned.columns.tolist()

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed_np = imputer.fit_transform(train_df_aligned)
    X_test_imputed_np = imputer.transform(test_df_aligned)

    std_devs_after_impute = np.std(X_train_imputed_np, axis=0)
    constant_indices_1 = np.where(std_devs_after_impute == 0)[0]

    if len(constant_indices_1) > 0:
        print(f"DEBUG: Removing {len(constant_indices_1)} constant columns after imputation.")
        X_train_imputed_np = np.delete(X_train_imputed_np, constant_indices_1, axis=1)
        X_test_imputed_np = np.delete(X_test_imputed_np, constant_indices_1, axis=1)
        current_feature_names = [name for i, name in enumerate(current_feature_names) if i not in constant_indices_1]

    # If all columns became constant, current_feature_names would be empty
    if not current_feature_names:
        print("Warning: All features became constant after imputation and first cleaning step.")
        # Return empty structures or handle as error
        return {'X_train': X_train_imputed_np, 'y_train': train_target.values, 'X_test': X_test_imputed_np, 'y_test': test_target.values, 
                'feature_names': [], 'scaler': None, 'power_transformer': None, 'imputer': imputer, 
                'pca_subject_model': pca_subject_model, 'pca_behavior_model': pca_behavior_model}


    train_df_for_poly = pd.DataFrame(X_train_imputed_np, columns=current_feature_names, index=train_df_aligned.index)
    test_df_for_poly = pd.DataFrame(X_test_imputed_np, columns=current_feature_names, index=test_df_aligned.index)

    base_continuous_for_poly = ['weekly_absence', 'c_math', 'sc_math', 'sc_history', 'sc_physics',
                                'sc_chemistry', 'sc_biology', 'sc_english', 'sc_geography',
                                'health_rat', 'gaming_h', 'avg_subject_score', 'std_subject_score']
    pca_cols_current = [col for col in current_feature_names if 'pca_' in col]
    poly_candidate_cols = [col for col in base_continuous_for_poly + pca_cols_current if col in train_df_for_poly.columns]

    train_df_after_poly = train_df_for_poly.copy()
    test_df_after_poly = test_df_for_poly.copy()

    if poly_candidate_cols:
        # print(f"DEBUG: Columns for polynomial features: {poly_candidate_cols}")
        poly_train_generated = add_polynomial_features_v2(train_df_for_poly[poly_candidate_cols], degree=2, interaction_only=True)
        poly_test_generated = add_polynomial_features_v2(test_df_for_poly[poly_candidate_cols], degree=2, interaction_only=True)

        if not poly_train_generated.empty:
             train_df_after_poly = pd.concat([train_df_after_poly.reset_index(drop=True), poly_train_generated.reset_index(drop=True)], axis=1)
        if not poly_test_generated.empty:
             test_df_after_poly = pd.concat([test_df_after_poly.reset_index(drop=True), poly_test_generated.reset_index(drop=True)], axis=1)
    else:
        print("Warning: No columns selected for polynomial features during preprocessing.")
        
    train_df_after_poly = train_df_after_poly.loc[:, ~train_df_after_poly.columns.duplicated()]
    test_df_after_poly = test_df_after_poly.loc[:, ~test_df_after_poly.columns.duplicated()]

    final_common_cols_poly = list(set(train_df_after_poly.columns) & set(test_df_after_poly.columns))
    if not final_common_cols_poly:
        print("Error: No common columns after polynomial feature step. Aborting preprocessing.")
        return {'X_train': np.array([]), 'y_train': np.array([]), 'X_test': np.array([]), 'y_test': np.array([]), 
                'feature_names': [], 'scaler': None, 'power_transformer': None, 'imputer': imputer, 
                'pca_subject_model': pca_subject_model, 'pca_behavior_model': pca_behavior_model}


    train_df_final = train_df_after_poly[sorted(final_common_cols_poly)].reset_index(drop=True)
    test_df_final = test_df_after_poly[sorted(final_common_cols_poly)].reset_index(drop=True)
    
    current_feature_names = train_df_final.columns.tolist()

    # Shuffle train_df_final and align train_target accordingly
    # Use the original index of train_features_raw for train_target before shuffling
    # If train_df_final's index is reset, we need to map y_train to the shuffled X_train
    original_train_indices = train_features_raw.index # Assuming train_features_raw had a meaningful index for y
    # If train_target was loaded aligned with train_features_raw:
    # y_train_temp = train_target.loc[original_train_indices] 

    # The current train_df_final has a reset index from 0 to N-1
    # So, when we shuffle it, its index will be shuffled.
    # We need to shuffle y_train based on the original indices that correspond to the rows in train_df_final
    # This is complex if indices were dropped and reset multiple times without care.
    # Safest: ensure y_train (train_target) is shuffled consistently with X_train (train_df_final)
    # X_train_df_shuffled = train_df_final.sample(frac=1, random_state=42)
    # y_train_series_shuffled = train_target.iloc[X_train_df_shuffled.index].reset_index(drop=True) 
    # ^ This assumes train_target's original index aligns with train_df_final's original index before shuffling.

    # Let's assume train_target has the same length and original order as train_features_raw
    # And train_df_final rows still correspond to train_features_raw rows before shuffling
    shuffled_indices = train_df_final.sample(frac=1, random_state=42).index # These are 0 to N-1
    X_train_df_shuffled = train_df_final.iloc[shuffled_indices]
    y_train_series_shuffled = train_target.iloc[shuffled_indices] # Shuffle y using the same integer indices

    X_test_df_ordered = test_df_final


    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train_df_shuffled)
    X_test_scaled_np = scaler.transform(X_test_df_ordered)

    std_devs_after_scale = np.std(X_train_scaled_np, axis=0)
    constant_indices_2 = np.where(std_devs_after_scale == 0)[0]

    if len(constant_indices_2) > 0:
        print(f"DEBUG: Removing {len(constant_indices_2)} constant columns after scaling.")
        X_train_scaled_np = np.delete(X_train_scaled_np, constant_indices_2, axis=1)
        X_test_scaled_np = np.delete(X_test_scaled_np, constant_indices_2, axis=1)
        current_feature_names = [name for i, name in enumerate(current_feature_names) if i not in constant_indices_2]

    # If all columns are removed, X_train_scaled_np will be empty (shape[1]==0)
    if X_train_scaled_np.shape[1] == 0:
        print("Warning: All features removed after scaling and second constant cleaning step.")
        return {'X_train': X_train_scaled_np, 'y_train': y_train_series_shuffled.values, 'X_test': X_test_scaled_np, 'y_test': test_target.values, 
                'feature_names': [], 'scaler': scaler, 'power_transformer': None, 'imputer': imputer, 
                'pca_subject_model': pca_subject_model, 'pca_behavior_model': pca_behavior_model}


    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    X_train_transformed_np = power_transformer.fit_transform(X_train_scaled_np)
    X_test_transformed_np = power_transformer.transform(X_test_scaled_np)

    final_feature_names_list = current_feature_names

    return {
        'X_train': X_train_transformed_np,
        'y_train': y_train_series_shuffled.values,
        'X_test': X_test_transformed_np,
        'y_test': test_target.values,
        'feature_names': final_feature_names_list,
        'scaler': scaler, 'power_transformer': power_transformer, 'imputer': imputer,
        'pca_subject_model': pca_subject_model, 'pca_behavior_model': pca_behavior_model
    }
