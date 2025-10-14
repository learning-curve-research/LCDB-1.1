from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import openml
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse import lil_matrix
import time
import matplotlib.pyplot as plt
import json
from os import path
import logging

from directencoder import DirectEncoder    # local


logger = logging.getLogger('lcdb')

# might be useful for naive bayes learner in the future
def mark_feature_type(X_feature, y_label):
    # check feature type
    dtypes = pd.concat([X_feature, y_label], axis=1).dtypes

    categorical = dtypes[(dtypes == 'object') | (dtypes == 'category')].index.tolist()
    numerical = dtypes[~((dtypes == 'object') | (dtypes == 'category'))].index.tolist()
    try:
        numerical.remove(y_label.name)
    except ValueError:
        pass
    try:
        categorical.remove(y_label.name)
    except ValueError:
        pass
    # print("Numerical features:", numerical)
    # print("Categorical features:", categorical)
    return numerical, categorical


def config_prepipeline(X_feature, minmax_scaler, mix_encode=False, onehot_threshold=1):

    ### check feature type
    # dtypes = pd.concat([X_feature, y_label], axis=1).dtypes
    dtypes = X_feature.dtypes
    categorical = dtypes[(dtypes == 'object') | (dtypes == 'category')].index.tolist()
    numerical = dtypes[~((dtypes == 'object') | (dtypes == 'category'))].index.tolist()

    ### Initialize pipelines for numerical and categorical features
    if minmax_scaler:
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('to_array', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),   # for  sparse data
            ('scaler', MinMaxScaler()),  ###### minmax scaling ######
            # ('scaler', StandardScaler()),  ###### standard scaling case #######
        ])
    else: 
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('to_array', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),   # for  sparse data
        ])

    # Create a dictionary to store the transformers for each categorical column
    transformers = []

    for col in categorical:
        if mix_encode:
            raise RuntimeError("The 'mix_encode' logic should not be activated.")
        else:
            transformers.append((f'onehot_{col}', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                # ('scaler', StandardScaler()),     ###### standard scaling case #######
            ]), [col]))

    # Combine numerical and categorical pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        [('num', numerical_pipeline, numerical),] + transformers)

    # Create pipeline with preprocessing and scaling
    preprocess_pipeline = Pipeline([
        ('preprocessor', preprocessor)
        ])
    
    return preprocess_pipeline


def prepipeline_naive_bayes(X_feature, minmax_scaler=False):
    
    dtypes = X_feature.dtypes
    categorical = dtypes[(dtypes == 'object') | (dtypes == 'category')].index.tolist()
    numerical = dtypes[~((dtypes == 'object') | (dtypes == 'category'))].index.tolist()

    categorical_indices = [(col in categorical) for col in X_feature.columns]

    if minmax_scaler:
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('to_array', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),   # for  sparse data
            ('scaler', MinMaxScaler()),     ###### minmax scaling #######
            # ('scaler', StandardScaler()),   ###### standard scaling case #######
        ])
    else: 
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('to_array', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),   # for  sparse data
        ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        # ('scaler', StandardScaler()),     ###### standard scaling case #######
        ('imputer_for_unseen', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical),
            ('cat', categorical_pipeline, categorical)
        ])

    preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])
    
    return preprocess_pipeline, categorical_indices


def prepipeline_numerical_only(X_feature, scaler = False):

    dtypes = X_feature.dtypes
    categorical = dtypes[(dtypes == 'object') | (dtypes == 'category')].index.tolist()
    numerical = dtypes[~((dtypes == 'object') | (dtypes == 'category'))].index.tolist()

    if scaler:
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('to_array', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),   # for  sparse data
            ('scaler', MinMaxScaler()),     ###### minmax scaling #######
            # ('scaler', StandardScaler()),   ###### standard scaling case #######
        ])
    else: 
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('to_array', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),   # for  sparse data
        ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))  
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical),
            ('cat', categorical_pipeline, categorical)
        ])

    preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])

    return preprocess_pipeline



def get_dataset(openmlid, feature_scaling, mix, preprocess=True): 
    '''
    feature_scaling: True / False
    mix: mix onehot and ordinal encoding True / False
    '''
    ###### load dataset ######
    dataset = openml.datasets.get_dataset(openmlid, download_data=True, download_qualities=True)
    # Fetch the data and target (features and labels)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if preprocess==False:
        print(f"Loading raw data from OpenML ID {openmlid}")
        return X, y

    ###### preprocessing and rescaling ######
    pipeline = config_prepipeline(X, minmax_scaler=feature_scaling, mix_encode=mix)

    # fit and transform
    X_preprocessed = pipeline.fit_transform(X)

    # if sparse list contained in array
    if any(issparse(matrix) for row in X_preprocessed for matrix in row):
        dense_matrices = np.array([[matrix.toarray() if issparse(matrix) else matrix for matrix in row] for row in X_preprocessed])
    else: 
        dense_matrices = X_preprocessed

    # removing columns with exactly the same values
    X_clean = dense_matrices[:, ~np.all(dense_matrices == dense_matrices[0, :], axis=0)]

    if feature_scaling: 
        print(f"Loading cleaned data from OpenML ID {openmlid} with feature scaling")
    else: 
        print(f"Loading cleaned data from OpenML ID {openmlid} without feature scaling")
    return X_clean, y


def get_dataset_naive_bayes(openmlid, feature_scaling, preprocess=True):
    '''
    feature_scaling: True / False
    '''

    dataset = openml.datasets.get_dataset(openmlid, download_data=True, download_qualities=True)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)


    pipeline, categorical_indices = prepipeline_naive_bayes(X, minmax_scaler=feature_scaling)

    if preprocess == False:
        print(f"Loading raw data from OpenML ID {openmlid}")
        return X, y, categorical_indices
    
    X_preprocessed = pipeline.fit_transform(X)

    # sparse case
    if issparse(X_preprocessed):
        X_preprocessed = X_preprocessed.toarray()

    # don't drop the same value column for maintain the indices of categorical column correct
    # X_clean = X_preprocessed[:, ~np.all(X_preprocessed == X_preprocessed[0, :], axis=0)]

    if feature_scaling: 
        print(f"Loading cleaned data from OpenML ID {openmlid} with feature scaling")
    else: 
        print(f"Loading cleaned data from OpenML ID {openmlid} without feature scaling")
        
    return X_preprocessed, y, categorical_indices


def get_dataset_cate(openmlid, preprocess=True):
    dataset = openml.datasets.get_dataset(openmlid, download_data=True, download_qualities=True)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if not preprocess:
        print(f"Loading raw data from OpenML ID {openmlid}")
        return X, y

    # missing imputer
    for col in X.select_dtypes(include=['object', 'category']):
        X[col] = X[col].fillna(X[col].mode()[0])
    for col in X.select_dtypes(exclude=['object', 'category']):
        X[col] = X[col].fillna(X[col].median())

    print(f"Loading cleaned data from OpenML ID {openmlid} without feature scaling")
    return X, y


def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_outer_split(X, y, seed):
    test_samples_at_90_percent_training = int(X.shape[0] * 0.1)
    if test_samples_at_90_percent_training <= 5000:
        return train_test_split(X, y, train_size = 0.9, random_state=seed, stratify=y)
    else:
        return train_test_split(X, y, train_size = X.shape[0] - 5000, test_size = 5000, random_state=seed, stratify=y)

def get_inner_split(X, y, outer_seed, inner_seed):
    X_learn, X_test, y_learn, y_test = get_outer_split(X, y, outer_seed)
    
    validation_samples_at_90_percent_training = int(X_learn.shape[0] * 0.1)
    if validation_samples_at_90_percent_training <= 5000:
        X_train, X_valid, y_train, y_valid = train_test_split(X_learn, y_learn, train_size = 0.9, random_state=inner_seed, stratify=y_learn)
    else:
        logger.info(f"Creating sample with instances: {X_learn.shape[0] - 5000}")
        X_train, X_valid, y_train, y_valid = train_test_split(X_learn, y_learn, train_size = X_learn.shape[0] - 5000, test_size = 5000, random_state=inner_seed, stratify=y_learn)
                                                                                      
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed):
    if issparse(y):
        y = y.toarray()
    y = np.ravel(y)
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(X, y, outer_seed, inner_seed)
    if anchor > X_train.shape[0]:
        raise ValueError(f"Invalid anchor {anchor} when available training instances are only {X_train.shape[0]}.")
    return X_train[:anchor], X_valid, X_test, y_train[:anchor], y_valid, y_test


def get_truth_and_predictions(learner_inst, X, y, anchor, outer_seed=0, inner_seed=0, 
                                realistic=False, fs_realistic=True, input_mode='normal'):
    """
    input_mode = 'normal' or 'mixNB' or 'cate_input'
    """

    # create a random split based on the seed
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed)

    # fit the model
    start_time = time.time()

    if realistic and input_mode == 'mixNB': 
        prepipeline, cate_indices = prepipeline_naive_bayes(X_train, minmax_scaler=fs_realistic)
        pipeline_inst = Pipeline([
                ('preprocessor', prepipeline), 
                ('learner', learner_inst),    # overwrite previous cate_idx
                ])
    elif realistic and input_mode == 'normal': 
        pipeline_inst = Pipeline([
                ('preprocessor', config_prepipeline(X_train, minmax_scaler = fs_realistic, mix_encode=False)), 
                ('learner', learner_inst),
                ])
    elif realistic and input_mode == 'cate_input': 
        pipeline_inst = Pipeline([
                ('preprocessor', prepipeline_numerical_only(X_train, scaler = fs_realistic)), 
                ('learner', learner_inst),
                ])
    elif realistic and input_mode == 'cate_input_tabnet': 
        pipeline_inst = Pipeline([
                ('learner', learner_inst),
                ]) 
    else: 
        pipeline_inst = Pipeline([
                ('learner', learner_inst),
                ]) 

    pipeline_inst.fit(X_train, y_train)

    train_time = time.time() - start_time

    logger.info(f"Training ready. Obtaining predictions for {X_test.shape[0]} instances.")

    # compute predictions on train data
    start_time = time.time()
    y_hat_train = pipeline_inst.predict(X_train)
    predict_time_train = time.time() - start_time

    # compute predictions on validation data
    start_time = time.time()
    y_hat_valid = pipeline_inst.predict(X_valid)
    predict_time_valid = time.time() - start_time
    start_time = time.time()

    if hasattr(pipeline_inst, 'predict_proba'):
        try: 
            y_prob_valid = pipeline_inst.predict_proba(X_valid)
        except: 
            y_prob_valid = None
    elif hasattr(pipeline_inst, 'decision_function'):
        try:
            y_prob_valid = pipeline_inst.decision_function(X_valid)
        except: 
            y_prob_valid = None
    else:
        y_prob_valid = None
    
    predict_proba_time_valid = time.time() - start_time

    # compute predictions on test data
    start_time = time.time()
    y_hat_test = pipeline_inst.predict(X_test)
    predict_time_test = time.time() - start_time
    start_time = time.time()

    if hasattr(pipeline_inst, 'predict_proba'):
        try: 
            y_prob_test = pipeline_inst.predict_proba(X_test)
        except: 
            y_prob_test = None
    elif hasattr(pipeline_inst, 'decision_function'):
        try:
            y_prob_test = pipeline_inst.decision_function(X_test)
        except: 
            y_prob_test = None
    else:
        y_prob_test = None
        
    predict_proba_time_test = time.time() - start_time

    # return all information
    return (y_train, y_valid, y_test, 
            y_hat_train, # y_prob_train, 
            y_hat_valid, y_prob_valid, 
            y_hat_test, y_prob_test, 
            learner_inst.classes_, 
            train_time, predict_time_train, # predict_proba_time_train, 
            predict_time_valid, predict_proba_time_valid, 
            predict_time_test, predict_proba_time_test, 
            )

def get_entry_learner(learner_name, learner_params, X, y, anchor, outer_seed, inner_seed, realistic, fs_realistic, encoder = DirectEncoder()):
    
    # get learner
    if learner_name == 'catboost': 
        from catboost import CatBoostClassifier
    
        dtypes = X.dtypes
        categorical = dtypes[(dtypes == 'object') | (dtypes == 'category')].index.tolist()
        numerical = dtypes[~((dtypes == 'object') | (dtypes == 'category'))].index.tolist()

        # categorical_indices = [i for i, col in enumerate(X.columns) if col in categorical]
        categorical_indices = [len(numerical) + i for i in range(len(categorical))]

        # replace None 
        for c in categorical:
            X[c] = X[c].astype(str).replace({None: 'none', 'None': 'none'})
            
        learner_inst = CatBoostClassifier(cat_features = categorical_indices, random_state = 42, task_type = "CPU", verbose = 0)

        (y_train, y_valid, y_test, y_hat_train, y_hat_valid, y_prob_valid, y_hat_test, y_prob_test, 
        known_labels, train_time, predict_time_train, predict_time_valid, 
        predict_proba_time_valid, predict_time_test, predict_proba_time_test
        ) = get_truth_and_predictions(learner_inst, X, y, anchor, outer_seed, inner_seed, realistic, fs_realistic, input_mode='cate_input')

    elif learner_name == 'tabnet':
        from pytorch_tabnet.tab_model import TabNetClassifier
        
        dtypes = X.dtypes
        categorical = dtypes[(dtypes == 'object') | (dtypes == 'category')].index.tolist()

        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        X[categorical] = cat_encoder.fit_transform(X[categorical].astype(str))

        categorical_indices = [i for i, col in enumerate(X.columns) if col in categorical]
        categorical_dim = [len(categories) for categories in cat_encoder.categories_]

        learner_inst = TabNetClassifier( cat_idxs = categorical_indices, cat_dims = categorical_dim, verbose = 0 )

        (y_train, y_valid, y_test, y_hat_train, y_hat_valid, y_prob_valid, y_hat_test, y_prob_test, 
        known_labels, train_time, predict_time_train, predict_time_valid, 
        predict_proba_time_valid, predict_time_test, predict_proba_time_test
        ) = get_truth_and_predictions(learner_inst, X, y, anchor, outer_seed, inner_seed, realistic, fs_realistic, input_mode='cate_input')
        
    else: 
        learner_class = get_class(learner_name)
        learner_inst = learner_class(**learner_params)
    
        # run learner
        (y_train, y_valid, y_test, y_hat_train, y_hat_valid, y_prob_valid, y_hat_test, y_prob_test, 
        known_labels, train_time, predict_time_train, predict_time_valid, 
        predict_proba_time_valid, predict_time_test, predict_proba_time_test
        ) = get_truth_and_predictions(learner_inst, X, y, anchor, outer_seed, inner_seed, realistic, fs_realistic)

    # print("Val accuracy: ",accuracy_score(y_valid , y_hat_valid))
    # print("Test accuracy: ",accuracy_score(y_test , y_hat_test))
    
    # compute info entry
    info = {
        "size_train": anchor,
        "size_test": len(y_test),
        "outer_seed": outer_seed,
        "inner_seed": inner_seed,
        "traintime": np.round(train_time, 4),
        "labels": [str(l) for l in known_labels],
        "y_train": encoder.encode_label_vector_compression(y_train),
        "y_valid": encoder.encode_label_vector_compression(y_valid),
        "y_test": encoder.encode_label_vector_compression(y_test),
        "y_hat_train": encoder.encode_label_vector_compression(y_hat_train),
        "y_hat_valid": encoder.encode_label_vector_compression(y_hat_valid),
        "y_hat_test": encoder.encode_label_vector_compression(y_hat_test),
        # "predictproba_train": encoder.encode_distribution(y_prob_train),
        # "predictproba_train_compressed": encoder.encode_distribution_compression(y_prob_train),
        # "predictproba_valid": encoder.encode_distribution(y_prob_valid),
        "predictproba_valid_compressed": encoder.encode_distribution_compression(y_prob_valid),
        # "predictproba_test": encoder.encode_distribution(y_prob_test),
        "predictproba_test_compressed": encoder.encode_distribution_compression(y_prob_test),
        "predicttime_train": np.round(predict_time_train, 4),
        # "predicttimeproba_train": np.round(predict_proba_time_train, 4),
        "predicttime_valid": np.round(predict_time_valid, 4),
        "predicttimeproba_valid": np.round(predict_proba_time_valid, 4),
        "predicttime_test": np.round(predict_time_test, 4),
        "predicttimeproba_test": np.round(predict_proba_time_test, 4)
    }
    
    return info
