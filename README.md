# LCDB 1.1: LCDB 1.1: A Database Illustrating Learning Curves are more Ill-Behaved than Previously Thought
### Project Structure
```
📦 Project Root
├── 📂 analysis — Analysis notebooks
├── 📂 dataset — The precomputed LCDB 1.1 datasets 
├── 📂 experiments — Raw experimental results (from which LCDB 1.1 can be derrived)
├── 📂 lcdb_function — Functions for computing the learning curves
└── Files for organising the large-scale experiments
```

### Updated LCDB 1.1 Features 
| Database   | Data Acquisition             | Feature Scaling | Estimation Anchor               | Learner \| Dataset  |
|------------|--------------------------|-----------------|---------------------------------|----------------------|
| **LCDB 1.0**  | with data-leakage (dl)    | none              | ⌈ 16 ⋅ 2<sup>n/2</sup> ⌉               | 20 \| 196  |
| **LCDB 1.1**  | with and without dl   | none, min-max, standard        | 4 times denser                 | 28 \| 265    |

### Installation for analysis
1. Download the precomputed learning curves from [here](https://surfdrive.surf.nl/files/index.php/s/4PEosYYoiHwB6uy) and extract them to the folder `dataset`.
2. Install the packages in `analysis/requirements.txt`. 
3. For the existing analysis notebooks, it is assumed that `latex` and several fonts, and `dvipng` is installed. Install with:

```
sudo apt install dvipng
sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super
```
   
4. The jupyter notebooks in the `analysis` folder illustrate the useage. For a simple example, see `database - shape.ipynb`.

### How to use the LCDB 1.1?
[Readme](./dataset/README.md) in dataset folder and refer to [LCDB11_viz.ipynb](./dataset/LCDB11_viz.ipynb) notebook. Useful metafeatures can be found in [`meta_feature.py`](./analysis/meta_feature.py)

### Contact us if you need more metrics!
We have stored all the probabilistic outputs and / or scores (when available for the learner). However, these files are too large for us to host for the broader public. If you are interested in a particular metric that we do not include, please contact us and we can compute it and host it. 

### Workflow for recomputing the LCDB 1.1
To ensure the reproducibility, please download the container [here](https://surfdrive.surf.nl/files/index.php/s/TSe0nqWKcT5jPwK)

1. Ensure there are three folder under path `/lcdb1.1`: `jobs`, `logs`, `results`. 
2. Run jobs_create.ipynb to create the jobs of generating learning curves with different setting combinations. 
3. Submit the jobs through `job_container.sh`. 
4. Check the status of jobs and create jobs for resubmition by `jobs_resubmit.ipynb`. 
5. Rename the `results` folder as `results1`, `results2`, ..., and create a new `results` folder
6. Repeat step 3-5 until all jobs are done. 
7. Use `jobs_gen_database.py` to generate the dataset. 



