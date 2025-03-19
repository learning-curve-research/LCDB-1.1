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
# from tqdm import tqdm
# import shutil, tarfile
import json
from os import path
# import importlib.resources as pkg_resources
# from io import StringIO
import logging

from lcdb_function.directencoder import DirectEncoder    # local


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


def config_prepipeline(X_feature, minmax_scaler, mix_encode=False, onehot_threshold=2):

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
            # ('scaler', MinMaxScaler()),  # minmax not support sparse data
            ('scaler', StandardScaler()),  ###### standard scaling case #######
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
            # unique_values = X_feature[col].nunique()
            # if unique_values > onehot_threshold and minmax_scaler == True:
            #     transformers.append((f'ord_{col}', Pipeline([
            #         ('imputer', SimpleImputer(strategy='most_frequent')),
            #         ('ordinal_encoder', OrdinalEncoder()),
            #         ('scaler', MinMaxScaler()),  
            #     ]), [col]))
            # elif unique_values > onehot_threshold and minmax_scaler == False:
            #     transformers.append((f'ord_{col}', Pipeline([
            #         ('imputer', SimpleImputer(strategy='most_frequent')),
            #         ('ordinal_encoder', OrdinalEncoder())
            #     ]), [col]))
            # else:
            #     transformers.append((f'onehot_{col}', Pipeline([
            #         ('imputer', SimpleImputer(strategy='most_frequent')),
            #         ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            #     ]), [col]))
        else:
            transformers.append((f'onehot_{col}', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                ('scaler', StandardScaler()),     ###### standard scaling case #######
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
            # ('scaler', MinMaxScaler()),  
            ('scaler', StandardScaler()),   ###### standard scaling case #######
        ])
    else: 
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('to_array', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),   # for  sparse data
        ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('scaler', StandardScaler()),     ###### standard scaling case #######
        ('imputer_for_unseen', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical),
            ('cat', categorical_pipeline, categorical)
        ])

    preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])
    
    return preprocess_pipeline, categorical_indices


def get_dataset(openmlid, feature_scaling, mix, preprocess=True): 
    '''
    feature_scaling: True / False
    mix: mix onehot and ordinal encoding True / False
    '''
    ###### load dataset ######
    ##########################
    dataset = openml.datasets.get_dataset(openmlid, download_data=True, download_qualities=True)
    # Fetch the data and target (features and labels)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if preprocess==False:
        print(f"Loading raw data from OpenML ID {openmlid}")
        return X, y

    ###### preprocessing and rescaling ######
    #########################################
    pipeline = config_prepipeline(X, minmax_scaler=feature_scaling, mix_encode=mix)

    # fit and transform
    X_preprocessed = pipeline.fit_transform(X)

    # taking care of sparse array
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

def get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed, monotonicity):
    if issparse(y):
        y = y.toarray()
    y = np.ravel(y)

    # check how seeds should be disturbed by monotonicity
    outer_seed_modifier = 0 if monotonicity in ["anchor", "inner", "outer"] else (1 + inner_seed)  # if we are not monotonic at outer level, use the inner seed to disturb the outer seed
    inner_seed_modifier = 0 if monotonicity in ["anchor", "inner"] else (1 + anchor)  # if we are not monotonic at inner level, use the anchor to disturb the inner seed

    X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(X, y, outer_seed + outer_seed_modifier, inner_seed + inner_seed_modifier)

    if anchor > X_train.shape[0]:
        raise ValueError(f"Invalid anchor {anchor} when available training instances are only {X_train.shape[0]}.")
    
    if monotonicity == "anchor":
        indices = range(anchor)
    else:
        indices = sorted(np.random.RandomState(anchor).choice(range(len(X_train)), anchor, replace=False))
    return X_train.iloc[indices] if isinstance(X_train, pd.DataFrame) else X_train[indices], X_valid, X_test, y_train[indices], y_valid, y_test
    #return X_train[:anchor], X_valid, X_test, y_train[:anchor], y_valid, y_test
    #else:
        #print("AOFHOSUHOSUGHSOIUHOSDFH")
        #X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(X, y, outer_seed, inner_seed + anchor)
        #X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(X, y, outer_seed, inner_seed)
        
    


def get_truth_and_predictions(
        learner_inst,
        X,
        y,
        anchor,
        outer_seed=0,
        inner_seed=0,
        realistic=False,
        fs_realisic=True,
        mixNB=False,
        monotonicity="anchor",
        verbose=False
        ):

    # create a random split based on the seed
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed, monotonicity=monotonicity)

    # fit the model
    start_time = time.time()
    if verbose:
        logger.info(f"Training {learner_inst} on data of shape {X_train.shape} using outer seed {outer_seed} and inner seed {inner_seed}")
    
    if realistic and mixNB: 
        prepipeline, cate_indices = prepipeline_naive_bayes(X_train, minmax_scaler=fs_realisic)
        pipeline_inst = Pipeline([
                ('preprocessor', prepipeline), 
                ('learner', learner_inst),    # overwrite previous cate_idx
                ])
    elif realistic and not mixNB: 
        pipeline_inst = Pipeline([
            ('preprocessor', config_prepipeline(X_train, minmax_scaler = fs_realisic, mix_encode=False)), 
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

def get_schedule(n):
    schedule = []
    i = 0
    while True:
        a = int(np.ceil(16 * 2 ** (i / 8)))
        if a > n:
            break
        schedule.append(a)
        i += 1
    return schedule

def get_entry_learner(
        learner_name,
        learner_params,
        X,
        y,
        anchor,
        outer_seed,
        inner_seed,
        realistic,
        fs_realisic,
        monotonicity,
        encoder = DirectEncoder(),
        verbose=False
        ):
    
    # get learner
    learner_class = get_class(learner_name)
    learner_inst = learner_class(**learner_params)
    
    # run learner
    (y_train, y_valid, y_test, y_hat_train, y_hat_valid, y_prob_valid, y_hat_test, y_prob_test, 
    known_labels, train_time, predict_time_train, predict_time_valid, 
    predict_proba_time_valid, predict_time_test, predict_proba_time_test
    ) = get_truth_and_predictions(
        learner_inst=learner_inst,
        X=X,
        y=y,
        anchor=anchor,
        outer_seed=outer_seed,
        inner_seed=inner_seed,
        realistic=realistic,
        fs_realisic=fs_realisic,
        mixNB=False,
        monotonicity=monotonicity,
        verbose=verbose
        )
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
    
    # show stats if desired            
    if verbose and info["predictproba_test"] is not None:
        y_test_after_compression = encoder.decode_distribution(info["predictproba_test"])
        log_loss_orig = np.round(metrics.log_loss(y_test, y_prob_test, labels=known_labels), 3)
        log_loss_after_compression = np.round(metrics.log_loss(y_test, y_test_after_compression, labels=known_labels), 3)
        diff = np.round(log_loss_orig - log_loss_after_compression, 3)
        gap = np.linalg.norm(y_test_after_compression - y_prob_test)
        logger.info(f"Proba-Matrix Gap: {gap}")
        logger.info(f"Change in log-loss due to compression: {diff} = {log_loss_orig} - {log_loss_after_compression}")
        
        if False:
            fig, ax = plt.subplots(1, y_prob_test.shape[1], figsize=(15, 5))
            for i, col in enumerate(y_prob_test.T):
                ax[i].scatter(col, y_test_after_compression[:,i], s=20)
                ax[i].plot([0,1], [0,1], linestyle="--", linewidth=1, color="black")
                ax[i].set_xscale("log")
                ax[i].set_yscale("log")
                ax[i].set_xlim([0.0001, 10])
                ax[i].set_ylim([0.0001, 10])
            plt.show()
    
    return info

def get_curve_for_metric_as_dataframe(json_curve_descriptor, metric, encoder = DirectEncoder(), error="raise", precision=4):
    
    cols = ['size_train', 'size_test', 'outer_seed', 'inner_seed', 'traintime']
    rows = []
    for entry in json_curve_descriptor:
        try:
            scores = get_metric_of_entry(entry, metric, encoder)
            if scores is None:
                row = [entry["size_train"], entry["size_test"], entry["outer_seed"], entry["inner_seed"], entry["traintime"], None, None, None]
            else:
                m_train, m_valid, m_test = scores
                row = [entry["size_train"], entry["size_test"], entry["outer_seed"], entry["inner_seed"], entry["traintime"], np.round(m_train, precision), np.round(m_valid, precision), np.round(m_test, precision)]
            rows.append(row)
        except:
            if error == "message":
                logger.info("Ignoring entry with exception!")
            else:
                raise
    return pd.DataFrame(rows, columns=cols + ["score_" + i for i in ["train", "valid", "test"]])


def get_curve_from_dataframe(curve_df):
    
    # sanity check to see that, if we know the learner, that there is only one of them
    if "learner" in curve_df.columns and len(pd.unique(curve_df["learner"])) > 1:
        raise Exception("pass only dataframes with entries for a single learner.")
    if "openmlid" in curve_df.columns and len(pd.unique(curve_df["openmlid"])) > 1:
        raise Exception("pass only dataframes with entries for a single openmlid.")
    
    # gather data
    anchors = sorted(list(pd.unique(curve_df["size_train"])))
    values_train = []
    values_valid = []
    values_test = []
    
    # extract curve
    for anchor in anchors:
        curve_df_anchor = curve_df[curve_df["size_train"] == anchor]
        values_train.append(list(curve_df_anchor["score_train"]))
        values_valid.append(list(curve_df_anchor["score_valid"]))
        values_test.append(list(curve_df_anchor["score_test"]))
        
    return anchors, values_train, values_valid, values_test

def get_curve_by_metric_from_json(json_curve_descriptor, metric, encoder = DirectEncoder(), error="raise"):
    
    return get_curve_from_dataframe(get_curve_for_metric_as_dataframe(json_curve_descriptor, metric, encoder, error))

        
        
def get_metric_of_entry(entry, metric, encoder = DirectEncoder()):
    y_train = encoder.decode_label_vector(entry["y_train"])
    y_valid = encoder.decode_label_vector(entry["y_valid"])
    y_test = encoder.decode_label_vector(entry["y_test"])
    labels = sorted(np.unique(y_train))
    
    if type(metric) == str:
        if metric in ["accuracy", "f1"]:
            y_hat_train = encoder.decode_label_vector(entry["y_hat_train"])
            y_hat_valid = encoder.decode_label_vector(entry["y_hat_valid"])
            y_hat_test = encoder.decode_label_vector(entry["y_hat_test"])
            
            keywords = {}
            if metric == "accuracy":
                m = metrics.accuracy_score
            elif metric == "f1":
                minority_class = labels[np.argmin(np.count_nonzero([y_train == l for l in labels]))]
                m = metrics.f1_score
                keywords["pos_label"] = minority_class
                if len(labels) > 2:
                    return None
                
            return m(y_train, y_hat_train, **keywords), m(y_valid, y_hat_valid, **keywords), m(y_test, y_hat_test, **keywords)
        
        elif metric in ["logloss", "auc"]:
            if entry["predictproba_train"] is not None:
                y_prob_train = encoder.decode_distribution(entry["predictproba_train"])
                y_prob_valid = encoder.decode_distribution(entry["predictproba_valid"])
                y_prob_test = encoder.decode_distribution(entry["predictproba_test"])
            else:
                y_hat_train = encoder.decode_label_vector(entry["y_hat_train"])
                y_hat_valid = encoder.decode_label_vector(entry["y_hat_valid"])
                y_hat_test = encoder.decode_label_vector(entry["y_hat_test"])
                labels = entry["labels"]
                y_prob_train = np.zeros((len(y_hat_train), len(labels)))
                for i, label in enumerate(y_hat_train):
                    y_prob_train[i,labels.index(label)] = 1
                y_prob_valid = np.zeros((len(y_hat_valid), len(labels)))
                for i, label in enumerate(y_hat_valid):
                    y_prob_valid[i,labels.index(label)] = 1
                y_prob_test = np.zeros((len(y_hat_test), len(labels)))
                for i, label in enumerate(y_hat_test):
                    y_prob_test[i,labels.index(label)] = 1
            
            if metric == "logloss":
                m = metrics.log_loss
                return m(y_train, y_prob_train, labels=entry["labels"]), m(y_valid, y_prob_valid, labels=entry["labels"]), m(y_test, y_prob_test, labels=entry["labels"])
            elif metric == "auc":
                
                if len(labels) > 2:
                    return None
                
                m = metrics.roc_auc_score
                return m(y_train, y_prob_train[:, 1], labels=entry["labels"]), m(y_valid, y_prob_valid[:, 1], labels=entry["labels"]), m(y_test, y_prob_test[:, 1], labels=entry["labels"])
            
            raise Exception(f"Unkown metric {metric}")
            
        raise Exception(f"Unknown metric {metric}")
    else:
        raise Exception("Currently only pre-defined metrics are supported.")



def plot_curve(anchors, points, ax, color, label = None):
    ax.plot(anchors, [np.median(v) for v in points], color=color, label=label)
    ax.plot(anchors, [np.mean(v) for v in points], linestyle="--", color=color)
    ax.fill_between(anchors, [np.percentile(v, 0) for v in points], [np.percentile(v, 100) for v in points], alpha=0.1, color=color)
    ax.fill_between(anchors, [np.percentile(v, 25) for v in points], [np.percentile(v, 75) for v in points], alpha=0.2, color=color)

def plot_train_and_test_curve(curve, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    anchors = curve[0]
    plot_curve(anchors, curve[1], ax, "C0", label="Performance on Training Data") # train curve
    plot_curve(anchors, curve[2], ax, "C1", label="Performance on Validation Data") # validation curve
    plot_curve(anchors, curve[3], ax, "C2", label="Performance on Test Data") # test curve
    
    ax.plot(anchors, [(np.mean(v_train) + np.mean(curve[2][a])) / 2 for a, v_train in enumerate(curve[1])], linestyle="--", color="black",linewidth=1)
    
    ax.axhline(np.mean(curve[2][-1]), linestyle="dotted", color="black",linewidth=1)
    ax.fill_between(anchors, np.mean(curve[2][-1]) - 0.0025, np.mean(curve[2][-1]) + 0.0025, color="black", alpha=0.1, hatch=r"//")
    
    ax.legend()
    ax.set_xlabel("Number of training instances")
    ax.set_ylabel("Prediction Performance")
    
    if fig is not None:
        return fig
        
