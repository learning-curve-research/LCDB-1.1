import numpy as np
import h5py
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from pathlib import Path
import sys
sys.path.append('lcdb_function')
from directencoder import DirectEncoder
from analysis.meta_feature import dataset_ids_CC18, dataset_ids_FULL, anchor_list_denser, learner_zoo_full_24, learner_zoo_full_mixNB


# ==================== Manual selection of metric ====================
# ====================================================================
# dataset list
dataset_ids = dataset_ids_CC18

# anchor definition
anchor_list = anchor_list_denser

# extracted learner list
learner_zoo = learner_zoo_full_24

# results folder index
start_index, end_index = 0, 14

# metric for extracting
metric = 'error rate'   # 'l1', 'log-loss', 'AUC'

# retrieval folder path
# input_path = '/lcdb11_raw/experiments_CC18_24_nofs_minmax/'
input_path = '/lcdb11/experiments_CC18_24_standard/'
print('Extracting from the path: ',Path(input_path))

# database file name
output_filename = "LCDB11_ER_CC18_standard.hdf5"

# min-max: 1, standardization: 2
hardcode_fs_channel = 2
# ====================================================================
# ====================================================================

# database setting: (dataset, learner, outer seed, inner seed, curve, train-val-test, feature scaling, clean-raw)
results_array = np.full((len(dataset_ids), len(learner_zoo), 5, 5, 137, 3, 3, 2), np.nan)  


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


def get_auc_from_json(json_obj):
    myEncoder = DirectEncoder()
    y_train = myEncoder.decode_label_vector_decompression(json_obj['y_train'])
    y_valid = myEncoder.decode_label_vector_decompression(json_obj['y_valid'])
    y_test = myEncoder.decode_label_vector_decompression(json_obj['y_test'])

    y_hat_proba_train = myEncoder.decode_distribution_decompress(json_obj['y_hat_proba_train'])
    y_hat_proba_valid = myEncoder.decode_distribution_decompress(json_obj['y_hat_proba_valid'])
    y_hat_proba_test = myEncoder.decode_distribution_decompress(json_obj['y_hat_proba_test'])

    auc_train = roc_auc_score(y_train, y_hat_proba_train, multi_class='ovr', average='weighted')
    auc_valid = roc_auc_score(y_valid, y_hat_proba_valid, multi_class='ovr', average='weighted')
    auc_test = roc_auc_score(y_test, y_hat_proba_test, multi_class='ovr', average='weighted')

    return auc_train, auc_valid, auc_test


def get_f1_from_json(json_obj):
    myEncoder = DirectEncoder()
    y_train = myEncoder.decode_label_vector_decompression(json_obj['y_train'])
    y_valid = myEncoder.decode_label_vector_decompression(json_obj['y_valid'])
    y_test = myEncoder.decode_label_vector_decompression(json_obj['y_test'])
    y_hat_train = myEncoder.decode_label_vector_decompression(json_obj['y_hat_train'])
    y_hat_valid = myEncoder.decode_label_vector_decompression(json_obj['y_hat_valid'])
    y_hat_test = myEncoder.decode_label_vector_decompression(json_obj['y_hat_test'])

    f1_train = f1_score(y_train, y_hat_train, average='weighted')
    f1_valid = f1_score(y_valid, y_hat_valid, average='weighted')
    f1_test = f1_score(y_test, y_hat_test, average='weighted')

    return f1_train, f1_valid, f1_test


def get_logloss_from_json(json_obj):
    myEncoder = DirectEncoder()
    y_train = myEncoder.decode_label_vector_decompression(json_obj['y_train'])
    y_valid = myEncoder.decode_label_vector_decompression(json_obj['y_valid'])
    y_test = myEncoder.decode_label_vector_decompression(json_obj['y_test'])

    y_hat_proba_train = myEncoder.decode_distribution_decompress(json_obj['y_hat_proba_train'])
    y_hat_proba_valid = myEncoder.decode_distribution_decompress(json_obj['y_hat_proba_valid'])
    y_hat_proba_test = myEncoder.decode_distribution_decompress(json_obj['y_hat_proba_test'])

    logloss_train = log_loss(y_train, y_hat_proba_train, labels=np.unique(y_train))
    logloss_valid = log_loss(y_valid, y_hat_proba_valid, labels=np.unique(y_valid))
    logloss_test = log_loss(y_test, y_hat_proba_test, labels=np.unique(y_test))

    return logloss_train, logloss_valid, logloss_test


for i in range(start_index, end_index):
    print(f"Extracting results from folder: {i}")
    for j in tqdm(range(1000)):
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
                            idx_dataset = np.digitize(json_obj['openmlid'], dataset_ids) - 1
                            # anchor index
                            idx_anchor = np.digitize(json_obj['size_train'], anchor_list) - 1
                            # learner index
                            idx_learner = learner_zoo.index(json_obj['learner'])
                            # outer and inner seed
                            outer = json_obj['outer_seed']
                            inner = json_obj['inner_seed']
                            # feature scaling
                            fs_channel = hardcode_fs_channel if json_obj['feature_scaling'] else 0
                            # raw clean
                            raw_clean_channel = 0 if json_obj['realistic'] else 1

                            if metric == 'error rate':
                                score_train, score_valid, score_test = get_acc_from_json(json_obj)
                                # Invert accuracy to error rate
                                score_train, score_valid, score_test = 1 - score_train, 1 - score_valid, 1 - score_test
                            elif metric == 'AUC':
                                score_train, score_valid, score_test = get_auc_from_json(json_obj)
                            elif metric == 'F1':
                                score_train, score_valid, score_test = get_f1_from_json(json_obj)
                            elif metric == 'log-loss':
                                score_train, score_valid, score_test = get_logloss_from_json(json_obj)
                            else:
                                raise ValueError(f"Unknown metric: {metric}")
                            acc_train, acc_valid, acc_test = get_acc_from_json(json_obj)

                            results_array[idx_dataset, idx_learner, outer, inner, idx_anchor, :, fs_channel, raw_clean_channel] = [score_train, score_valid, score_test]
                        
                        except Exception as e:
                            print(f"Error processing entry: {e}")
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in {file_path}: {e}")
        except FileNotFoundError:
            print(f"{file_path} is missing")


# store the database in hdf5
with h5py.File(output_filename, 'w') as f:
    # learning curve database
    print(f"Saving metric: {metric}")
    f.create_dataset(metric, data=results_array)
    # meta-data
    f.create_dataset('anchor list', data=np.array(anchor_list))
    f.create_dataset('dataset ids', data=np.array(dataset_ids))
    f.create_dataset('learner zoo', data=np.string_(learner_zoo))
print(f"{output_filename} successfully saved")