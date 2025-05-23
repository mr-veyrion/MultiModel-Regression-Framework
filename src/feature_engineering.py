
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

def advanced_feature_engineering_v2(df_input, is_train=True, pca_subject_model=None, pca_behavior_model=None):
    df = df_input.copy()

    if 'gpa_score' in df.columns: df = df.drop(columns=['gpa_score'])

    subject_score_cols = ['c_math', 'sc_math', 'sc_history', 'sc_physics',
                          'sc_chemistry', 'sc_biology', 'sc_english', 'sc_geography']
    behavioral_cols = ['weekly_absence', 'health_rat', 'gaming_h', 'sports',
                       'romantic', 'extracurri', 'part_time', 'gender']

    all_expected_cols = subject_score_cols + behavioral_cols
    for col in all_expected_cols:
        if col not in df.columns:
            df[col] = 0 

    for subj_col in subject_score_cols:
        if subj_col in df.columns:
            if 'gaming_h' in df.columns:
                df[f'{subj_col}_x_gaming'] = df[subj_col] * df['gaming_h']
            if 'health_rat' in df.columns:
                df[f'{subj_col}_x_health'] = df[subj_col] * df['health_rat']
            if 'weekly_absence' in df.columns:
                df[f'{subj_col}_x_absence'] = df[subj_col] * df['weekly_absence']

    if 'health_rat' in df.columns and 'gaming_h' in df.columns:
        df['health_x_gaming'] = df['health_rat'] * df['gaming_h']
    if 'extracurri' in df.columns and 'sports' in df.columns:
        df['extracurri_x_sports'] = df['extracurri'] * df['sports']

    present_subject_cols = [col for col in subject_score_cols if col in df.columns]
    if present_subject_cols:
        df['avg_subject_score'] = df[present_subject_cols].mean(axis=1)
        df['std_subject_score'] = df[present_subject_cols].std(axis=1).fillna(0)
        if 'avg_subject_score' in df.columns and 'health_rat' in df.columns:
             df['avg_score_x_health'] = df['avg_subject_score'] * df['health_rat']

    # PCA on Subject Scores
    present_subject_cols_for_pca = [col for col in subject_score_cols if col in df.columns]
    if len(present_subject_cols_for_pca) > 1: 
        subject_data_raw = df[present_subject_cols_for_pca].fillna(0)
        # Ensure columns passed to nunique are present
        non_constant_subject_cols = [col for col in subject_data_raw.columns if subject_data_raw[col].nunique(dropna=False) > 1]

        if len(non_constant_subject_cols) >= 1: 
            subject_data_for_pca = subject_data_raw[non_constant_subject_cols]
            n_components_subject = min(3, subject_data_for_pca.shape[1])
            if n_components_subject > 0:
                if is_train:
                    pca_subject_model = PCA(n_components=n_components_subject, random_state=42)
                    try:
                        subject_pca_features = pca_subject_model.fit_transform(subject_data_for_pca)
                        for i in range(subject_pca_features.shape[1]):
                            df[f'pca_subject_{i+1}'] = subject_pca_features[:, i]
                    except ValueError as e:
                        print(f"FE: PCA fit on subject scores failed: {e}")
                        pca_subject_model = None 
                elif pca_subject_model and hasattr(pca_subject_model, 'transform') and hasattr(pca_subject_model, 'n_features_in_'):
                    try:
                        cols_for_transform_subj = pca_subject_model.feature_names_in_
                        if all(c in df.columns for c in cols_for_transform_subj):
                            subject_data_test_pca = df[cols_for_transform_subj].fillna(0)
                            subject_pca_features = pca_subject_model.transform(subject_data_test_pca)
                            for i in range(subject_pca_features.shape[1]):
                                df[f'pca_subject_{i+1}'] = subject_pca_features[:, i]
                        else:
                            # print("FE: PCA subject transform: Not all features PCA was trained on are present in test data.")
                            for i in range(pca_subject_model.n_components_): df[f'pca_subject_{i+1}'] = 0
                    except Exception as e:
                        # print(f"FE: PCA transform on subject scores failed: {e}")
                        for i in range(pca_subject_model.n_components_ if pca_subject_model and hasattr(pca_subject_model, 'n_components_') else 0):
                             df[f'pca_subject_{i+1}'] = 0
            else:
                 if is_train: pca_subject_model = None
        else: 
            if is_train: pca_subject_model = None
            print("FE: Not enough non-constant subject features for PCA.")


    # PCA on Behavioral metrics
    present_behavior_cols_for_pca = [col for col in ['weekly_absence', 'health_rat', 'gaming_h'] if col in df.columns]
    if len(present_behavior_cols_for_pca) > 1:
        behavior_data_raw = df[present_behavior_cols_for_pca].fillna(0)
        non_constant_behavior_cols = [col for col in behavior_data_raw.columns if behavior_data_raw[col].nunique(dropna=False) > 1]

        if len(non_constant_behavior_cols) >=1:
            behavior_data_for_pca = behavior_data_raw[non_constant_behavior_cols]
            n_components_behavior = min(2, behavior_data_for_pca.shape[1])
            if n_components_behavior > 0:
                if is_train:
                    pca_behavior_model = PCA(n_components=n_components_behavior, random_state=123)
                    try:
                        behavior_pca_features = pca_behavior_model.fit_transform(behavior_data_for_pca)
                        for i in range(behavior_pca_features.shape[1]):
                            df[f'pca_behavior_{i+1}'] = behavior_pca_features[:, i]
                    except ValueError as e:
                        # print(f"FE: PCA fit on behavior scores failed: {e}")
                        pca_behavior_model = None
                elif pca_behavior_model and hasattr(pca_behavior_model, 'transform') and hasattr(pca_behavior_model, 'n_features_in_'):
                    try:
                        cols_for_transform_behav = pca_behavior_model.feature_names_in_
                        if all(c in df.columns for c in cols_for_transform_behav):
                            behavior_data_test_pca = df[cols_for_transform_behav].fillna(0)
                            behavior_pca_features = pca_behavior_model.transform(behavior_data_test_pca)
                            for i in range(behavior_pca_features.shape[1]):
                                df[f'pca_behavior_{i+1}'] = behavior_pca_features[:, i]
                        else:
                            # print("FE: PCA behavior transform: Not all features PCA was trained on are present in test data.")
                            for i in range(pca_behavior_model.n_components_): df[f'pca_behavior_{i+1}'] = 0
                    except Exception as e:
                        # print(f"FE: PCA transform on behavior scores failed: {e}")
                        for i in range(pca_behavior_model.n_components_ if pca_behavior_model and hasattr(pca_behavior_model, 'n_components_') else 0):
                            df[f'pca_behavior_{i+1}'] = 0
            else: 
                if is_train: pca_behavior_model = None
        else: 
            if is_train: pca_behavior_model = None
            print("FE: Not enough non-constant behavior features for PCA.")
    
    df = df.fillna(0) 

    if is_train:
        return df, pca_subject_model, pca_behavior_model
    else:
        return df


def add_polynomial_features_v2(X_df, degree=2, interaction_only=False):
    if X_df.empty or X_df.shape[1] == 0:
        return pd.DataFrame(index=X_df.index)
    
    numeric_cols = X_df.select_dtypes(include=np.number).columns
    if not numeric_cols.any():
        return pd.DataFrame(index=X_df.index)

    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interaction_only)
    X_poly_values = poly.fit_transform(X_df[numeric_cols])
    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    X_poly_df = pd.DataFrame(X_poly_values, columns=poly_feature_names, index=X_df.index)
    return X_poly_df
