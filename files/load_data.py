from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import numpy as np

def load_adnimerge_data():
    adnimerge = pd.read_parquet('../data/reduced_baseline_adnimerge_df.parquet', 'pyarrow')
    adnimerge.dropna(inplace=True)
    adnimerge['DX_bl'] = adnimerge['DX_bl'].astype('string')
    adnimerge_sub = adnimerge[adnimerge['DX_bl'].isin(['LMCI', 'EMCI'])] #binary problem
    adnimerge_sub = adnimerge_sub[['RID', 'DX_bl']]
    return adnimerge_sub
    

def load_lowdim_data(seed):
    adnimerge_sub = load_adnimerge_data()

    roi_aparc = pd.read_parquet('../data/baseline_roi_aparc_df.parquet')
    roi_aparc.drop(columns= ['lMedial_wall', 'rMedial_wall'], inplace=True)
    roi_aparc['RID'] = roi_aparc['RID'].astype(int)
    roi_aparc.drop(columns=['subject', 'session_number'], inplace=True)
    data = adnimerge_sub.merge(roi_aparc, on='RID', how='inner')
    
    y_classification = data['DX_bl']
    X = data.drop(columns=['RID', 'DX_bl'])
        
    X = X.astype('float32')
    y_clas_enc = LabelEncoder()
    y_classification = y_clas_enc.fit_transform(y_classification)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=seed
        )

    return X_train, X_test, y_train, y_test


def load_lowdim_neuro_data(seed):
    adnimerge_sub = load_adnimerge_data()

    roi = pd.read_parquet('../data/baseline_roi_neuromorph_df.parquet')
    roi['RID'] = roi['RID'].astype(int)
    roi.drop(columns=['subject', 'session_number'], inplace=True)
    data = adnimerge_sub.merge(roi, on='RID', how='inner')
    
    y_classification = data['DX_bl']
    X = data.drop(columns=['RID', 'DX_bl'])
        
    X = X.astype('float32')
    y_clas_enc = LabelEncoder()
    y_classification = y_clas_enc.fit_transform(y_classification)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=seed
        )

    return X_train, X_test, y_train, y_test




def load_highdim_data(seed):
    adnimerge_sub = load_adnimerge_data()

    hd_data = pd.read_parquet('../data/baseline_hd_df.parquet')
    hd_data.drop(columns=['ses'], inplace=True)
    hd_data['RID'] = hd_data['RID'].astype(int)
    data = adnimerge_sub.merge(hd_data, on='RID', how='inner')

    y_classification = data['DX_bl']

    # Convert the 'img' column into a 2D NumPy array
    feature_matrix = np.stack(data['img'].values)
    X_expanded = pd.DataFrame(feature_matrix, columns=[f'feature_{i}' for i in range(feature_matrix.shape[1])])
        
    X = X_expanded.astype('float32')
    y_clas_enc = LabelEncoder()
    y_classification = y_clas_enc.fit_transform(y_classification)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=seed
    )

    return X_train, X_test, y_train, y_test

