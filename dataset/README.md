# Introduction of Learning Curves Database 1.1 (LCDB 1.1)

### Different versions of the data:
The dataset names follow this format: `LCDB11_<Metric>_<Dataset>_<Learner>`
Each component represents different characteristics, explained as follows:
- **Metric**: 
    - `ER`: Error Rate
    - `AUC`: weighted One-vs-the-Rest (OvR) for multiclass 
    - `f1`: F1 score
    - `log-loss`: aka logistic loss or cross-entropy loss
    - `traintime`: training time of the model (in seconds)
    - `msg`: note that some of the values of learning curves may be missing and are stored as a NaN value. This array stores the error message to explain why the value is missing. 
- **Dataset**: The dataset we used for generating learning curves. Meta data about dataset is stored in folder [metadata](../metadata/). 
    - `CC-18`: the OpenML dataset `CC-18 Benchmark`, 72 datasets, see [`datasets_CC18_ids.csv`](../metadata/datasets_CC18_ids.csv). 
    - `265`: the superset of LCDB 1.0 and `CC-18`, 265 datasets, see [`datasets_FULL_ids.csv`](../metadata/datasets_FULL_ids.csv).
- **Learner**: we have two group here
    - `24` contains 24 learner, see [`learner_zoo.csv`](../metadata/learner_zoo.csv)
    - `4mixNB` contains 4 mix Naive Bayes methods, see [`learner_zoo_mixNB.csv`](../metadata/learner_zoo_mixNB.csv)
    - `catboost`, `realmlp`, `tabnet` contains no feature scaling with no data-leakage version
    - `tabpfn` contains 3 feature scaling with no data-leakage data

### Data structure: 
When loading the dataset using HDF5 (e.g., `LCDB11_ER_265_24.hdf5`), you will get the shape of file in for example `LCDB11_ER_ 265.hdf5`: (265, 24, 5, 5, 137, 3, 3, 2), each dimension corresponds to the following components, in order: 
`(dataset, learner, outer split, inner split, anchor, train-val-test, noFS-mixmaxFS-standardFS, noDataLeakage-DataLeakage), `
where the dimensions are:
  - `dataset`: index indicating the dataset, see [metadata](../metadata/) folder
  - `learner`: index indicating the learner, see [metadata](../metadata/) folder
  - `outer split` and `inner split`: an integer in [0,1,2,3,4]
  - `anchor`: index indicating $k$ of the training set size, where $k \in [0,1,2,...]$. The training set size is given by $n_k = \lceil 16 \cdot 2^{k/8} \rceil$, see [`anchor_list_denser.csv`](../metadata/anchor_list_denser.csv).
  - `train-val-test`: 
    - 0: training set
    - 1: validation set
    - 2: test set
  - `noFS-minmaxFS-standardFS`: 
    - 0: no feature scaling
    - 1: min-max feature scaling 
    - 2: standardization scaling
  - `noDataLeakage-DataLeakage`: 
    - 0: no-dataleakage version (preprocessor is fitted to train data only), 
    - 1: dataleakage version (preprocessor is fit on the complete dataset).