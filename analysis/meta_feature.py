import numpy as np

# CC-18
dataset_ids_CC18 = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 38, 44, 46, 50, 54, 151, 182, 188, 300, 307, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1461, 1462, 1464, 1468, 1475, 1478, 1480, 1485, 1486, 1487, 1489, 1494, 1497, 1501, 1510, 1590, 4134, 4534, 4538, 6332, 23381, 23517, 40499, 40668, 40670, 40701, 40923, 40927, 40966, 40975, 40978, 40979, 40982, 40983, 40984, 40994, 40996, 41027]
# FULL: 265
dataset_ids_FULL = [3, 6, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 36, 37, 38, 44, 46, 50, 54, 55, 57, 60, 61, 151, 179, 180, 181, 182, 184, 185, 188, 201, 273, 293, 299, 300, 307, 336, 346, 351, 354, 357, 380, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 446, 458, 469, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1000, 1002, 1018, 1019, 1020, 1021, 1036, 1040, 1041, 1042, 1049, 1050, 1053, 1056, 1063, 1067, 1068, 1069, 1083, 1084, 1085, 1086, 1087, 1088, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1233, 1235, 1236, 1441, 1448, 1450, 1457, 1461, 1462, 1464, 1465, 1468, 1475, 1477, 1478, 1479, 1480, 1483, 1485, 1486, 1487, 1488, 1489, 1494, 1497, 1499, 1501, 1503, 1509, 1510, 1515, 1566, 1567, 1575, 1590, 1592, 1597, 4134, 4135, 4137, 4534, 4538, 4541, 6332, 23381, 23512, 23517, 40498, 40499, 40664, 40668, 40670, 40672, 40677, 40685, 40687, 40701, 40713, 40900, 40910, 40923, 40927, 40966, 40971, 40975, 40978, 40979, 40981, 40982, 40983, 40984, 40994, 40996, 41027, 41142, 41143, 41144, 41145, 41146, 41150, 41156, 41157, 41158, 41159, 41161, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41228, 41972, 42734, 42742, 42769, 42809, 42810]

# memory_openml_ids = [1111, 1112, 1114, 42732, 42733]
# dataset_ids_270 = [3, 6, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 36, 37, 38, 44, 46, 50, 54, 55, 57, 60, 61, 151, 179, 180, 181, 182, 184, 185, 188, 201, 273, 293, 299, 300, 307, 336, 346, 351, 354, 357, 380, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 446, 458, 469, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1000, 1002, 1018, 1019, 1020, 1021, 1036, 1040, 1041, 1042, 1049, 1050, 1053, 1056, 1063, 1067, 1068, 1069, 1083, 1084, 1085, 1086, 1087, 1088, 1111, 1112, 1114, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1233, 1235, 1236, 1441, 1448, 1450, 1457, 1461, 1462, 1464, 1465, 1468, 1475, 1477, 1478, 1479, 1480, 1483, 1485, 1486, 1487, 1488, 1489, 1494, 1497, 1499, 1501, 1503, 1509, 1510, 1515, 1566, 1567, 1575, 1590, 1592, 1597, 4134, 4135, 4137, 4534, 4538, 4541, 6332, 23381, 23512, 23517, 40498, 40499, 40664, 40668, 40670, 40672, 40677, 40685, 40687, 40701, 40713, 40900, 40910, 40923, 40927, 40966, 40971, 40975, 40978, 40979, 40981, 40982, 40983, 40984, 40994, 40996, 41027, 41142, 41143, 41144, 41145, 41146, 41150, 41156, 41157, 41158, 41159, 41161, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41228, 41972, 42732, 42733, 42734, 42742, 42769, 42809, 42810]

anchor_list_denser = np.ceil(16 * 2 ** ((np.arange(137)) / 8)).astype(int)

anchor_list_lcdb10 = np.ceil(16 * 2 ** ((np.arange(35)) / 2)).astype(int)

# import sys
# sys.path.append("../")
# from lcdb_function.lcdb import get_dataset
# column_counts = []
# class_counts = []
# for dataset_id in dataset_ids:
#     X, y = get_dataset(dataset_id, feature_scaling=False, mix=False, preprocess=True)
#     num_class = len(set(y))
    # # Remove columns where all rows have the same value across the column
    # unique_rows = np.unique(X, axis=1)
    # column_counts.append(unique_rows.shape[1])
#     class_counts.append(num_class)
# column_counts_str = ', '.join(map(str, column_counts))
# class_counts_str = ', '.join(map(str, class_counts))
feature_num_CC18 = [73, 16, 4, 216, 76, 9, 64, 6, 47, 24, 62, 46, 61, 16, 8, 51, 57, 287, 27, 18, 14, 36, 91, 617, 27, 70, 21, 719, 37, 37, 21, 21, 21, 21, 51, 4, 4, 856, 51, 561, 11, 500, 174, 72, 5, 41, 24, 256, 30, 105, 1776, 68, 32, 160, 156, 21, 40, 126, 360, 33, 784, 3072, 77, 21, 3113, 240, 27, 5, 16, 18, 784, 6]
feature_num_CC18_remove_redundancy = [73, 16, 4, 213, 76, 9, 64, 6, 47, 24, 62, 43, 61, 16, 8, 51, 57, 255, 27, 18, 14, 36, 63, 617, 27, 70, 21, 718, 37, 37, 21, 21, 21, 21, 51, 4, 4, 694, 50, 540, 11, 500, 152, 72, 5, 41, 24, 256, 30, 105, 1776, 68, 32, 158, 155, 21, 37, 126, 360, 33, 784, 3072, 76, 21, 1525, 240, 27, 5, 16, 18, 784, 6]
class_num_CC18 = [2, 26, 3, 10, 10, 2, 10, 10, 10, 3, 10, 2, 2, 10, 2, 2, 2, 3, 2, 4, 2, 6, 5, 26, 11, 4, 6, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9, 6, 6, 2, 2, 2, 2, 2, 2, 4, 10, 2, 2, 2, 2, 5, 2, 2, 2, 11, 3, 3, 2, 46, 10, 8, 4, 2, 10, 7, 2, 7, 2, 10, 3]

learner_zoo_full_24 = [ 'SVC_linear', 'SVC_poly', 'SVC_rbf', 'SVC_sigmoid', 'sklearn.tree.DecisionTreeClassifier', 'sklearn.tree.ExtraTreeClassifier', 'sklearn.linear_model.LogisticRegression', 'sklearn.linear_model.PassiveAggressiveClassifier', 'sklearn.linear_model.Perceptron', 'sklearn.linear_model.RidgeClassifier', 'sklearn.linear_model.SGDClassifier', 'sklearn.neural_network.MLPClassifier', 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis', 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis', 'sklearn.naive_bayes.BernoulliNB', 'sklearn.naive_bayes.MultinomialNB', 'sklearn.naive_bayes.ComplementNB', 'sklearn.naive_bayes.GaussianNB', 'sklearn.neighbors.KNeighborsClassifier', 'sklearn.neighbors.NearestCentroid', 'sklearn.ensemble.ExtraTreesClassifier', 'sklearn.ensemble.RandomForestClassifier', 'sklearn.ensemble.GradientBoostingClassifier', 'sklearn.dummy.DummyClassifier' ]
learner_zoo = [ 'SVM_Linear', 'SVM_Poly', 'SVM_RBF', 'SVM_Sigmoid', 'Decision Tree', 'ExtraTree','LogisticRegression', 'PassiveAggressive', 'Perceptron', 'RidgeClassifier', 'SGDClassifier', 'MLP', 'LDA', 'QDA', 'BernoulliNB', 'MultinomialNB', 'ComplementNB', 'GaussianNB','KNN', 'NearestCentroid', 'ens.ExtraTrees', 'ens.RandomForest', 'ens.GradientBoosting','DummyClassifier']
learner_zoo_abbreviation = [ 'SVM_Linear', 'SVM_Poly', 'SVM_RBF', 'SVM_Sigmoid', 'DT', 'ET','LR', 'PA', 'Perceptron', 'Ridge', 'SGD', 'MLP', 'LDA', 'QDA', 'BernoulliNB', 'MultinomialNB', 'ComplementNB', 'GaussianNB','KNN', 'NC', 'ens.ET', 'ens.RF', 'ens.GB','Dummy']

learner_zoo_mixNB = [ "mixBernoulliNB", "mixMultinomialNB", "mixComplementNB", "mixGaussianNB"]