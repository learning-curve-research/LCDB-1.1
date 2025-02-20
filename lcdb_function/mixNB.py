import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.naive_bayes import CategoricalNB
import logging
# local
from lcdb import get_truth_and_predictions
from directencoder import DirectEncoder 

logger = logging.getLogger('lcdb')

class MixedNBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, categorical_mask, unfitted_nb_numerical):
        self.categorical_mask = np.array(categorical_mask)
        self.unfitted_nb_categorical = CategoricalNB()
        self.unfitted_nb_numerical = unfitted_nb_numerical
        self.fitted_nb_numerical = None
        self.fitted_nb_categorical = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        assert(len(self.categorical_mask) == X.shape[1])

        X_cat = X[:,self.categorical_mask==True]
        X_num = X[:,self.categorical_mask==False]
        
        # all feature could be numerical
        if X_cat.shape[1] > 0:
            self.fitted_nb_categorical = clone(self.unfitted_nb_categorical)
            self.fitted_nb_categorical.fit(X_cat, y)
        # same, all feature could be categorical
        if X_num.shape[1] > 0:
            self.fitted_nb_numerical = clone(self.unfitted_nb_numerical)
            self.fitted_nb_numerical.fit(X_num, y)

        if self.fitted_nb_categorical and self.fitted_nb_numerical:
            assert (self.fitted_nb_categorical.classes_ == self.fitted_nb_numerical.classes_).all()
            self.classes_ = self.fitted_nb_categorical.classes_
        elif self.fitted_nb_categorical:
            self.classes_ = self.fitted_nb_categorical.classes_
        elif self.fitted_nb_numerical:
            self.classes_ = self.fitted_nb_numerical.classes_

        # Return the classifier
        return self
    
    def predict_log_proba(self, X):
        X_cat = X[:, self.categorical_mask == True]
        X_num = X[:, self.categorical_mask == False]

        # init log proba
        log_proba_cat = np.zeros((X.shape[0], len(self.classes_)))
        log_proba_num = np.zeros((X.shape[0], len(self.classes_)))

        # same as fit, consider the extreme case
        # there are a lot of encoded unseen categorical feature
        if self.fitted_nb_categorical:
            X_cat = np.asarray(X_cat, dtype=int)  # ensure integer

            # check the unseen encoded label
            for i in range(X_cat.shape[1]):
                max_index = self.fitted_nb_categorical.feature_log_prob_[i].shape[1]
                if np.any(X_cat[:, i] >= max_index): 
                    unseen_mask = X_cat[:, i] >= max_index
                    print(f"Warning: Found unseen categories in feature {i}.")
                    # log proba = -inf
                    log_proba_cat[unseen_mask, :] = -np.inf

            # exclude the -inf case
            valid_mask = ~np.isinf(log_proba_cat).any(axis=1)  
            if valid_mask.any():
                log_proba_cat[valid_mask, :] = self.fitted_nb_categorical.predict_log_proba(X_cat[valid_mask])

        if self.fitted_nb_numerical:
            log_proba_num = self.fitted_nb_numerical.predict_log_proba(X_num)

        # combine log probabilities 
        total_log_prob = log_proba_cat + log_proba_num
        if self.fitted_nb_categorical:
            total_log_prob -= self.fitted_nb_categorical.class_log_prior_

        return total_log_prob
    
    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        
        posterior = self.predict_proba(X)

        return self.classes_[np.argmax(posterior, axis=1)]


def get_entry_mixNB(learner_name, learner_params, X, y, anchor, outer_seed, inner_seed, realistic, fs_realisic, encoder = DirectEncoder(), verbose=False):
    
    # get learner
    learner_inst = MixedNBClassifier(**learner_params)
    
    # run learner
    (y_train, y_valid, y_test, y_hat_train, y_hat_valid, y_prob_valid, y_hat_test, y_prob_test, 
    known_labels, train_time, predict_time_train, predict_time_valid, 
    predict_proba_time_valid, predict_time_test, predict_proba_time_test
    ) = get_truth_and_predictions(learner_inst, X, y, anchor, outer_seed, inner_seed, realistic, fs_realisic, mixNB=True, verbose=verbose)
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