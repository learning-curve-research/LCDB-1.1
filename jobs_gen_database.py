import numpy as np
import h5py
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys
sys.path.append('lcdb_function')
from directencoder import DirectEncoder

# CC-18
dataset_ids = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 38, 44, 46, 50, 54, 151, 182, 188, 300, 307, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1461, 1462, 1464, 1468, 1475, 1478, 1480, 1485, 1486, 1487, 1489, 1494, 1497, 1501, 1510, 1590, 4134, 4534, 4538, 6332, 23381, 23517, 40499, 40668, 40670, 40701, 40923, 40927, 40966, 40975, 40978, 40979, 40982, 40983, 40984, 40994, 40996, 41027]
# 270
# dataset_ids_270 = [3, 6, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 36, 37, 38, 44, 46, 50, 54, 55, 57, 60, 61, 151, 179, 180, 181, 182, 184, 185, 188, 201, 273, 293, 299, 300, 307, 336, 346, 351, 354, 357, 380, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 446, 458, 469, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1000, 1002, 1018, 1019, 1020, 1021, 1036, 1040, 1041, 1042, 1049, 1050, 1053, 1056, 1063, 1067, 1068, 1069, 1083, 1084, 1085, 1086, 1087, 1088, 1111, 1112, 1114, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1233, 1235, 1236, 1441, 1448, 1450, 1457, 1461, 1462, 1464, 1465, 1468, 1475, 1477, 1478, 1479, 1480, 1483, 1485, 1486, 1487, 1488, 1489, 1494, 1497, 1499, 1501, 1503, 1509, 1510, 1515, 1566, 1567, 1575, 1590, 1592, 1597, 4134, 4135, 4137, 4534, 4538, 4541, 6332, 23381, 23512, 23517, 40498, 40499, 40664, 40668, 40670, 40672, 40677, 40685, 40687, 40701, 40713, 40900, 40910, 40923, 40927, 40966, 40971, 40975, 40978, 40979, 40981, 40982, 40983, 40984, 40994, 40996, 41027, 41142, 41143, 41144, 41145, 41146, 41150, 41156, 41157, 41158, 41159, 41161, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41228, 41972, 42732, 42733, 42734, 42742, 42769, 42809, 42810]
# FULL 265
# dataset_ids = [3, 6, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 36, 37, 38, 44, 46, 50, 54, 55, 57, 60, 61, 151, 179, 180, 181, 182, 184, 185, 188, 201, 273, 293, 299, 300, 307, 336, 346, 351, 354, 357, 380, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 446, 458, 469, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1000, 1002, 1018, 1019, 1020, 1021, 1036, 1040, 1041, 1042, 1049, 1050, 1053, 1056, 1063, 1067, 1068, 1069, 1083, 1084, 1085, 1086, 1087, 1088, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1233, 1235, 1236, 1441, 1448, 1450, 1457, 1461, 1462, 1464, 1465, 1468, 1475, 1477, 1478, 1479, 1480, 1483, 1485, 1486, 1487, 1488, 1489, 1494, 1497, 1499, 1501, 1503, 1509, 1510, 1515, 1566, 1567, 1575, 1590, 1592, 1597, 4134, 4135, 4137, 4534, 4538, 4541, 6332, 23381, 23512, 23517, 40498, 40499, 40664, 40668, 40670, 40672, 40677, 40685, 40687, 40701, 40713, 40900, 40910, 40923, 40927, 40966, 40971, 40975, 40978, 40979, 40981, 40982, 40983, 40984, 40994, 40996, 41027, 41142, 41143, 41144, 41145, 41146, 41150, 41156, 41157, 41158, 41159, 41161, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41228, 41972, 42734, 42742, 42769, 42809, 42810]

anchor_list = np.ceil(16 * 2 ** ((np.arange(137)) / 8)).astype(int)

# learner_zoo = [ 'SVC_linear',
#                 'SVC_poly',
#                 'SVC_rbf',
#                 'SVC_sigmoid',
#                 'sklearn.tree.DecisionTreeClassifier',
#                 'sklearn.tree.ExtraTreeClassifier',
#                 'sklearn.linear_model.LogisticRegression',
#                 'sklearn.linear_model.PassiveAggressiveClassifier',
#                 'sklearn.linear_model.Perceptron',
#                 'sklearn.linear_model.RidgeClassifier',
#                 'sklearn.linear_model.SGDClassifier',
#                 'sklearn.neural_network.MLPClassifier',
#                 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis',
#                 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis',
#                 'sklearn.naive_bayes.BernoulliNB',
#                 'sklearn.naive_bayes.MultinomialNB',
#                 'sklearn.naive_bayes.ComplementNB',
#                 'sklearn.naive_bayes.GaussianNB',
#                 'sklearn.neighbors.KNeighborsClassifier',
#                 'sklearn.neighbors.NearestCentroid',
#                 'sklearn.ensemble.ExtraTreesClassifier',
#                 'sklearn.ensemble.RandomForestClassifier',
#                 'sklearn.ensemble.GradientBoostingClassifier',
#                 'sklearn.dummy.DummyClassifier'
#                   ]

learner_zoo = [ "mixBernoulliNB", 
               "mixGaussianNB", 
               "mixMultinomialNB", 
               "mixComplementNB"]

def get_er_from_json(json_obj):

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

results_array_fs_clean = np.full((len(dataset_ids), len(learner_zoo), 5, 5, 137, 3), np.nan)
# results_array_nofs_clean = np.full((len(dataset_ids), len(learner_zoo), 5, 5, 137, 3), np.nan)
results_array_fs_real = np.full((len(dataset_ids), len(learner_zoo), 5, 5, 137, 3), np.nan)
# results_array_nofs_real = np.full((len(dataset_ids), len(learner_zoo), 5, 5, 137, 3), np.nan)

for i in range(0, 4): # 0-10
    print(f"Extracting result from file: ", i)
    for j in tqdm(range(0, 1000)):
        file_name = '/lcdb11_raw/experiments_270_mixNB_standard/results%d/result_%d.json' % (i, j)
        try: 
            with open(file_name, 'r') as file:
                # Read file line by line
                for line in file:
                    try:
                        # Deserialize the JSON object
                        json_obj = json.loads(line.strip())
                        # Check the value of the "status" key
                        if json_obj.get('status') == 'ok':
                            anchor = json_obj.get('size_train')
                            inner = json_obj.get('inner_seed')
                            outer = json_obj.get('outer_seed')
                            openmlid = json_obj.get('openmlid')
                            learner = json_obj.get('learner')

                            anchor_index = np.digitize(anchor, anchor_list) - 1
                            index_dataset = np.digitize(openmlid, dataset_ids) - 1
                            index_learner = learner_zoo.index(learner)

                            acc_train, acc_valid, acc_test = get_er_from_json(json_obj)

                            feature_scaling = json_obj.get('feature_scaling')
                            realistic = json_obj.get('realistic')
                            # print(f"fs{feature_scaling}, real{realistic}")

                            if feature_scaling and not realistic:
                                results_array_fs_clean[index_dataset, index_learner, outer, inner, anchor_index, 0] = 1 - acc_train
                                results_array_fs_clean[index_dataset, index_learner, outer, inner, anchor_index, 1] = 1 - acc_valid
                                results_array_fs_clean[index_dataset, index_learner, outer, inner, anchor_index, 2] = 1 - acc_test
                            
                            # elif not feature_scaling and not realistic:
                            #     results_array_nofs_clean[index_dataset, index_learner, outer, inner, anchor_index, 0] = 1 - acc_train
                            #     results_array_nofs_clean[index_dataset, index_learner, outer, inner, anchor_index, 1] = 1 - acc_valid
                            #     results_array_nofs_clean[index_dataset, index_learner, outer, inner, anchor_index, 2] = 1 - acc_test

                            elif feature_scaling and realistic:
                                results_array_fs_real[index_dataset, index_learner, outer, inner, anchor_index, 0] = 1 - acc_train
                                results_array_fs_real[index_dataset, index_learner, outer, inner, anchor_index, 1] = 1 - acc_valid
                                results_array_fs_real[index_dataset, index_learner, outer, inner, anchor_index, 2] = 1 - acc_test

                            # elif not feature_scaling and realistic:
                            #     results_array_nofs_real[index_dataset, index_learner, outer, inner, anchor_index, 0] = 1 - acc_train
                            #     results_array_nofs_real[index_dataset, index_learner, outer, inner, anchor_index, 1] = 1 - acc_valid
                            #     results_array_nofs_real[index_dataset, index_learner, outer, inner, anchor_index, 2] = 1 - acc_test


                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_name}: {e}")
        except: 
            print(f"{file_name} is missing")



# save as h5py
output_filename = "LCDB11_ER_CC18_standardFS_clean_mixNB.hdf5"
with h5py.File(output_filename, 'w') as f:
    f.create_dataset('error_rate', data = results_array_fs_clean)
print(f"{output_filename} successfully saved")

# output_filename = "LCDB11_ER_270_noFS_clean12.hdf5"
# with h5py.File(output_filename, 'w') as f:
#     f.create_dataset('error_rate', data = results_array_nofs_clean)
# print(f"{output_filename} successfully saved")

output_filename = "LCDB11_ER_CC18_standardFS_raw_mixNB.hdf5"
with h5py.File(output_filename, 'w') as f:
    f.create_dataset('error_rate', data = results_array_fs_real)
print(f"{output_filename} successfully saved")

# output_filename = "LCDB11_ER_270_noFS_raw12.hdf5"
# with h5py.File(output_filename, 'w') as f:
#     f.create_dataset('error_rate', data = results_array_nofs_real)
# print(f"{output_filename} successfully saved")