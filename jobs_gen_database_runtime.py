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


# ======================== Manual Settings ========================
# dataset list
dataset_ids = dataset_ids_FULL  #dataset_ids_CC18

# extracted learner list
learner_zoo = learner_zoo_full_24

# results folder index
start_index, end_index = 7, 10

# retrieval folder path
input_path = '/lcdb11_raw/experiments_198_24_standard/'     # jobs 1-29
# input_path = '/lcdb11_raw/experiments_198_mixNB_nofs_minmax/'     # jobs 1-4
# input_path = '/lcdb11_raw/experiments_270_mixNB_standard/'        # jobs 1-3
# input_path = '/lcdb11_raw/experiments_CC18_24_nofs_minmax/'         # jobs 1-12
input_path = '/lcdb11/experiments_198_24_nofs_minmax/'          # jobs 1-117
# input_path = '/lcdb11/experiments_CC18_24_standard/'        # jobs 1-10
# input_path = '/lcdb11/experiments_CC18_mixNB_nofs_minmax/'    # jobs 1-3

# database file name
output_filename = "LCDB11_ER_265_nofs_minmax_198_1.3.hdf5"


# ======================== Settings ========================
print(f'Extracting from the path: {Path(input_path)}')

# min-max: 1, standardization: 2
if 'minmax' in input_path.lower():
    hardcode_fs_channel = 1
elif 'standard' in input_path.lower():
    hardcode_fs_channel = 2
else:
    print("Error: Could not detect feature scaling type from input_path. Expected 'minmax' or 'standard'.")
    sys.exit(1)

######   NO train-val-test, 'ok' = 0, 'timeout' = 1, 'error' = error message
# database setting: (dataset, learner, outer seed, inner seed, curve, feature scaling, clean-raw)
results_array = np.full((len(dataset_ids), len(learner_zoo), 5, 5, 137, 3, 2), np.nan)  


# ======================== Run ========================
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

                        # encode into the index of dataset_ids list
                        idx_dataset = np.digitize(json_obj['openmlid'], dataset_ids) - 1
                        # anchor index
                        idx_anchor = np.digitize(json_obj['size_train'], anchor_list_denser) - 1
                        # learner index
                        idx_learner = learner_zoo.index(json_obj['learner'])
                        # outer and inner seed
                        outer = json_obj['outer_seed']
                        inner = json_obj['inner_seed']
                        # feature scaling
                        fs_channel = hardcode_fs_channel if json_obj['feature_scaling'] else 0
                        # raw clean
                        raw_clean_channel = 0 if json_obj['realistic'] else 1

                        results_array[idx_dataset, idx_learner, outer, inner, idx_anchor, fs_channel, raw_clean_channel] = json_obj['traintime']
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in {file_path}: {e}")
        except FileNotFoundError:
            print(f"{file_path} is missing")


# store the database in hdf5
with h5py.File(output_filename, 'w') as f:

    f.create_dataset('traintime', data=results_array)

    # meta-data
    f.create_dataset('anchor list', data=np.array(anchor_list_denser))
    f.create_dataset('dataset ids', data=np.array(dataset_ids))
    f.create_dataset('learner zoo', data=np.string_(learner_zoo))

print(f"{output_filename} successfully saved")
