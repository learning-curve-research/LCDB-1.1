### [Meeting logs](https://docs.google.com/document/d/1xjjYc7_MNnJEzLUTAif8uPNqt820LsviWb5zdeDC79U/edit?usp=sharing)

### Project Structure
```
📦 Project Root
├── 📂 analysis — Analysis notebooks
├── 📂 dataset — Usable datasets 
├── 📂 experiments — Raw experiment results
├── 📂 lcdb_function — Functions for computing
└── Files for Cluster Computing
```

### Installation for analysis

1. Download the precomputed learning curves from https://surfdrive.surf.nl/files/index.php/s/MtjBoStRcOZxJt6 and extract them to the folder `dataset`.
2. Install the packages in `analysis/requirements.txt`. 
3. For the existing analysis notebooks, it is assumed that `latex` and several fonts, and `dvipng` is installed. Install with:

```
sudo apt install dvipng
sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super
```
   
4. The jupyter notebooks in the `analysis` folder illustrate the useage. For a simple example, see `database - shape.ipynb`.

Different versions of the data:
- Cheng can you explain the filenames here? 

### Workflow for experiments
Previous: 
1. Ensure there are three folder under path `/lcdb1.1`: `jobs`, `logs`, `results`. 
2. Run jobs_create.ipynb to create the jobs of generating learning curves with different setting combinations. 
3. Submit the jobs through `job_container.sh`. 
4. Check the status of jobs and create jobs for resubmition by `jobs_resubmit.ipynb`. 
5. Rename the `results` folder as `results1`, `results2`, ..., and create a new `results` folder
6. Repeat step 3-5 until all jobs done. 
7. Use `jobs_gen_database.py` to generate the dataset. 

### Updated LCDB 1.1 Features 
| Database   | Imputation               | Feature Scaling | Estimation Anchor               | Learner \| Dataset  |
|------------|--------------------------|-----------------|---------------------------------|----------------------|
| **LCDB 1.0**  | globally                 | ❌              | ⌈ 16 ⋅ 2<sup>n/2</sup> ⌉               | 20 \| 220 (246) |
| **LCDB 1.1**  | train-only & globally   | ✅              | 4 times denser                 | 28 \| 265            |
### Useful Settings
##### Anchor
$S_{Anchor} = \lceil 16 \cdot 2^{k/d} \rceil$ where $k \in \{0, 1, 2, 3, ...\}$

LCDB 1.0: $d = 2$
```python
anchor_list = np.ceil(16 * 2 ** ((np.arange(35)) / 2)).astype(int)
```
LCDB 1.1: $d = 8$
```python
anchor_list = np.ceil(16 * 2 ** ((np.arange(137)) / 8)).astype(int)
```

##### learner
```python
learner_zoo = [ 'SVC_linear', 'SVC_poly', 'SVC_rbf', 'SVC_sigmoid', 'Decision Trees', 'ExtraTrees','LogisticRegression', 'PassiveAggressive', 'Perceptron', 'RidgeClassifier', 'SGDClassifier', 'MLP', 'LDA', 'QDA', 'BernoulliNB', 'MultinomialNB', 'ComplementNB', 'GaussianNB','KNN', 'NearestCentroid', 'ens.ExtraTrees', 'ens.RandomForest', 'ens.GradientBoosting','DummyClassifier']

```

##### Scikit-CC18 data ID (72)
```python
dataset_ids_CC18 = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 38, 44, 46, 50, 54, 151, 182, 188, 300, 307, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1461, 1462, 1464, 1468, 1475, 1478, 1480, 1485, 1486, 1487, 1489, 1494, 1497, 1501, 1510, 1590, 4134, 4534, 4538, 6332, 23381, 23517, 40499, 40668, 40670, 40701, 40923, 40927, 40966, 40975, 40978, 40979, 40982, 40983, 40984, 40994, 40996, 41027]
```
##### LCDB1.1 FULL data ID (265)
```python
dataset_ids_FULL = [3, 6, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 36, 37, 38, 44, 46, 50, 54, 55, 57, 60, 61, 151, 179, 180, 181, 182, 184, 185, 188, 201, 273, 293, 299, 300, 307, 336, 346, 351, 354, 357, 380, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 446, 458, 469, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1000, 1002, 1018, 1019, 1020, 1021, 1036, 1040, 1041, 1042, 1049, 1050, 1053, 1056, 1063, 1067, 1068, 1069, 1083, 1084, 1085, 1086, 1087, 1088, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1233, 1235, 1236, 1441, 1448, 1450, 1457, 1461, 1462, 1464, 1465, 1468, 1475, 1477, 1478, 1479, 1480, 1483, 1485, 1486, 1487, 1488, 1489, 1494, 1497, 1499, 1501, 1503, 1509, 1510, 1515, 1566, 1567, 1575, 1590, 1592, 1597, 4134, 4135, 4137, 4534, 4538, 4541, 6332, 23381, 23512, 23517, 40498, 40499, 40664, 40668, 40670, 40672, 40677, 40685, 40687, 40701, 40713, 40900, 40910, 40923, 40927, 40966, 40971, 40975, 40978, 40979, 40981, 40982, 40983, 40984, 40994, 40996, 41027, 41142, 41143, 41144, 41145, 41146, 41150, 41156, 41157, 41158, 41159, 41161, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41228, 41972, 42734, 42742, 42769, 42809, 42810]

```
