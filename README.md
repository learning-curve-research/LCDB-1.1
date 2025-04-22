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

1. Download the precomputed learning curves from [here](https://surfdrive.surf.nl/files/index.php/s/4PEosYYoiHwB6uy) and extract them to the folder `dataset`.
2. Install the packages in `analysis/requirements.txt`. 
3. For the existing analysis notebooks, it is assumed that `latex` and several fonts, and `dvipng` is installed. Install with:

```
sudo apt install dvipng
sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super
```
   
4. The jupyter notebooks in the `analysis` folder illustrate the useage. For a simple example, see `database - shape.ipynb`.

### Different versions of the data:

The dataset names follow this format:
```
LCDB11_<Metric>_<Dataset>
```

Each component represents different characteristics, explained as follows:
- **Metric**: 
    - `ER` stands for `Error Rate`.
    - `ACC` stands for `Accuracy`
    - `AUC`
    - `L1`
    - `LogLoss`
    - `ErrorMessage` stores the error message of missing point
- **Dataset**: The dataset we used for generating learning curves. 
    - `CC-18` indicates that the dataset `CC-18 Benchmark`, which consists of 72 datasets.
    - `265` indicated that the superset of the dataset in LCDB 1.0 and `CC-18`, which consists of 265 datasets. 

### Data structure: 
When loading the dataset using HDF5 (e.g., `LCDB11_ER_265.hdf5`), the data shape is:
If you load the database by hdf5, you will get the shape of file in for example `LCDB11_ER_ 265.hdf5`: (265, 24, 5, 5, 137, 3, 3, 2), each dimension corresponds to the following components, in order: 
```
(dataset, learner, outer split, inner split, anchor, train-val-test, noFS-mixmaxFS-standardFS, raw-clean), 
```
where
  - `noFS`: no feature scaling
  - `minmaxFS`: min-max feature scaling
  - `standardFS`: standardization feature scaling (z-score normalization)
  - `raw`: Preprocessing is performed only on the training set, simulating a more realistic scenario where data cleaning is required.
  - `clean`: Preprocessing is performed on the entire dataset, assuming access to clean data.


### Workflow for experiments
To ensure the reproducibility, please download the container [here](https://surfdrive.surf.nl/files/index.php/s/TSe0nqWKcT5jPwK)

1. Ensure there are three folder under path `/lcdb1.1`: `jobs`, `logs`, `results`. 
2. Run jobs_create.ipynb to create the jobs of generating learning curves with different setting combinations. 
3. Submit the jobs through `job_container.sh`. 
4. Check the status of jobs and create jobs for resubmition by `jobs_resubmit.ipynb`. 
5. Rename the `results` folder as `results1`, `results2`, ..., and create a new `results` folder
6. Repeat step 3-5 until all jobs done. 
7. Use `jobs_gen_database.py` to generate the dataset. 

### Updated LCDB 1.1 Features 
| Database   | Data Acquisition             | Feature Scaling | Estimation Anchor               | Learner \| Dataset  |
|------------|--------------------------|-----------------|---------------------------------|----------------------|
| **LCDB 1.0**  | clean          | ❌              | ⌈ 16 ⋅ 2<sup>n/2</sup> ⌉               | 20 \| 196 (246) |
| **LCDB 1.1**  | clean & raw   | ✅              | 4 times denser                 | 28 \| 265            |
### Useful Meta-Feature
See [`meta_feature.py`](./analysis/meta_feature.py)