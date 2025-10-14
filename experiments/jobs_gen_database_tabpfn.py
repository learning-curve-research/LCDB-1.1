import numpy as np
import h5py
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('lcdb_function')
from directencoder import DirectEncoder
from analysis.meta_feature import dataset_ids_tabpfn, dataset_ids_FULL, anchor_list_denser


# ======================== Manual Settings ========================
# dataset list
dataset_ids = dataset_ids_FULL

# results folder index
start_index, end_index = 1, 3

# metric for extracting
metric = 'error rate'  

# retrieval folder path
# input_path = '/lcdb11/experiments_265_RealMLP_nofs_minmax/'     # jobs 1-44
# input_path = '/lcdb11/experiments_265_TabPFN_nofs_minmax/'     # jobs 1-4
input_path = '/lcdb11/experiments_265_TabPFN_standard/'

# database file name
output_filename = "LCDB11_ER_265_tabpfn_standardfs.hdf5"


# ======================== Settings ========================
print(f'Extracting from the path: {Path(input_path)}')

# database setting: (dataset, outer seed, inner seed, curve, train-val-test)
results_array = np.full((len(dataset_ids), 5, 5, 137, 3), np.nan)  


# ======================== Metrics ========================
def get_acc_from_json(json_obj):
    myEncoder = DirectEncoder()
    y_train = myEncoder.decode_label_vector_decompression(json_obj['y_train'])
    y_valid = myEncoder.decode_label_vector_decompression(json_obj['y_valid'])
    y_test = myEncoder.decode_label_vector_decompression(json_obj['y_test'])
    y_hat_train = myEncoder.decode_label_vector_decompression(json_obj['y_hat_train'])
    y_hat_valid = myEncoder.decode_label_vector_decompression(json_obj['y_hat_valid'])
    y_hat_test = myEncoder.decode_label_vector_decompression(json_obj['y_hat_test'])
    acc_train = accuracy_score(y_train, y_hat_train)
    acc_valid = accuracy_score(y_valid, y_hat_valid)
    acc_test = accuracy_score(y_test, y_hat_test)
    return acc_train, acc_valid, acc_test


# ======================== Run ========================
for i in range(start_index, end_index):
    print(f"Extracting results from folder: {i}")
    
    for j in tqdm(range(1000)):
        non_nan_before = np.count_nonzero(~np.isnan(results_array))
        file_path = Path(input_path) / f'results{i}/result_{j}.json'
        try:
            with open(str(file_path), 'r') as file:
                for line in file:
                    try:
                        json_obj = json.loads(line.strip())
                        if json_obj.get('status') != 'ok':
                            continue

                        try:
                            # encode into the index of dataset_ids list
                            if json_obj['openmlid'] not in dataset_ids:
                                continue
                            idx_dataset = dataset_ids.index(json_obj['openmlid'])
                            # idx_dataset = np.digitize(json_obj['openmlid'], dataset_ids) - 1
                            # anchor index
                            idx_anchor = np.digitize(json_obj['size_train'], anchor_list_denser) - 1
                            # outer and inner seed
                            outer = json_obj['outer_seed']
                            inner = json_obj['inner_seed']

                            if metric == 'error rate':
                                score_train, score_valid, score_test = get_acc_from_json(json_obj)
                                # Invert accuracy to error rate
                                score_train, score_valid, score_test = 1 - score_train, 1 - score_valid, 1 - score_test
                            else:
                                raise ValueError(f"Unknown metric: {metric}")

                            # store the metrics that are successfully executed
                            results_array[idx_dataset, outer, inner, idx_anchor, :] = [score_train, score_valid, score_test]
                        
                        except Exception as e:
                            print(f"Error processing entry: {e}")
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in {file_path}: {e}")
        except FileNotFoundError:
            print(f"{file_path} is missing")
            
        non_nan_after = np.count_nonzero(~np.isnan(results_array))
        delta = non_nan_after - non_nan_before
        print(f"Folder {j} contributed {delta} new non-NaN values.")

# store the database in hdf5
with h5py.File(output_filename, 'w') as f:
    # learning curve database
    print(f"Saving metric: {metric}")
    f.create_dataset(metric, data=results_array)
    # meta-data
    # f.create_dataset('anchor list', data=np.array(anchor_list_denser))
    # f.create_dataset('dataset ids', data=np.array(dataset_ids))
print(f"{output_filename} successfully saved")