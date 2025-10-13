# Introduction of Learning Curves Database 1.1 (LCDB 1.1)

### Different versions of the data:
The dataset names follow this format: `LCDB11_<Metric>_<Dataset>_<Learner>`
Each component represents different characteristics, explained as follows:
- **Metric**: 
    - `ER` stands for Error Rate
    - `AUC` we use weighted One-vs-the-Rest (OvR) for multiclass 
    - `f1` stands for F1 score
    - `log-loss` aka logistic loss or cross-entropy loss
    - `traintime` refers to the training time of the model (in seconds)
    - `msg` note that some of the values of learning curves may be missing and are stored as a NaN value. This array stores the error message to explain why the value is missing. 
- **Dataset**: The dataset we used for generating learning curves. 
    - `CC-18` indicates that the dataset `CC-18 Benchmark`, which consists of 72 datasets. See `dataset_ids_CC18` in [`meta_feature.py`](../analysis/meta_feature.py)
    - `265` indicated that the superset of the dataset in LCDB 1.0 and `CC-18`, which consists of 265 datasets. See `dataset_ids_FULL` in [`meta_feature.py`](../analysis/meta_feature.py)
- **Learner**: we have two group here
    - `24` contains 24 learner, see `learner_zoo` in [`meta_feature.py`](../analysis/meta_feature.py)
    - `mixNB` contains 4 mix Naive Bayes methods, see `learner_zoo_mixNB` in [`meta_feature.py`](../analysis/meta_feature.py)

### Data structure: 
When loading the dataset using HDF5 (e.g., `LCDB11_ER_265_24.hdf5`), you will get the shape of file in for example `LCDB11_ER_ 265.hdf5`: (265, 24, 5, 5, 137, 3, 3, 2), each dimension corresponds to the following components, in order: 
`(dataset, learner, outer split, inner split, anchor, train-val-test, noFS-mixmaxFS-standardFS, noDataLeakage-DataLeakage), `
where the dimensions are:
  - `dataset`: index indicating the dataset, see [`meta_feature.py`](../analysis/meta_feature.py)
  - `learner`: index indicating the learner, see [`meta_feature.py`](../analysis/meta_feature.py)
  - `outer split` and `inner split`: an integer in [0,1,2,3,4]
  - `anchor`: index indicating $k$ of the training set size, where $k \in [0,1,2,...]$. The training set size is given by $n_k = \lceil 16 \cdot 2^{k/8} \rceil$.
  - `train-val-test`: 0 means train set, 1 means validation, 2 means test set.
  - `noFS-minmaxFS-standardFS`: 0 means no feature scaling, 1 means min-max feature scaling, 2 means features are standardized to have zero mean and a variance of 1.
  - `raw-clean`: 0 means no-dataleakage version (preprocessor is fitted to train data only), 1 means dataleakage version (preprocessor is fit on the complete dataset).