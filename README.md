# LCDB 1.1: A Database Illustrating Learning Curves are more Ill-Behaved than Previously Thought
### Project Structure
```
📦 Project Root
├── 📂 analysis — Analysis notebooks
├── 📂 dataset — The precomputed LCDB 1.1 datasets 
├── 📂 experiments — Files for organising the large-scale experiments
├── 📂 lcdb_function — Functions for computing the learning curves
└── LCDB11_demo.ipynb - showing how to use our database
```

### Updated LCDB 1.1 Features 
| Database   | Data Acquisition             | Feature Scaling | Estimation Anchor               | Learner \| Dataset  |
|------------|--------------------------|-----------------|---------------------------------|----------------------|
| **LCDB 1.0**  | with data-leakage (dl)    | none              | ⌈ 16 ⋅ 2<sup>n/2</sup> ⌉               | 20 \| 196  |
| **LCDB 1.1**  | with and without dl   | none, min-max, standard        | 4 times denser                 | 28 \| 265    |

### How to use the LCDB 1.1? 
First download the precomputed learning curves from [4TU.ResearchData](https://data.4tu.nl/private_datasets/V7dDlGyQJqPc_mXUAJL1MweACKG557GQtOWIVHhYpjU) and extract them to the folder [`dataset`](./dataset/). The [Readme](./dataset/README.md) in the dataset folder explains the files and their content. Or you can use the demonstration in the [`LCDB11_demo.ipynb`](./LCDB11_demo.ipynb) to download the data automatically. Useful metafeatures can be found in [`meta_feature.py`](./analysis/meta_feature.py).

### Reproduce all figures from the paper
Install the packages in [`analysis/requirements.txt`](./analysis/requirements.txt). Once the setup is complete,the analysis notebooks in the [`analysis`](./analysis/) folder illustrate the usage. 

### Contact us if you need more metrics!
We have stored all the probabilistic outputs and / or scores (when available for the learner). However, these files are too large for us to host for the broader public. If you are interested in a particular metric that we do not include, please contact us and we can compute it and host it. 

### Workflow for Computing LCDB 1.1
To ensure the reproducibility, please download the [container](https://surfdrive.surf.nl/files/index.php/s/TSe0nqWKcT5jPwK) (or generate it by using [`lcdb11container_builer.zip`](./experiments/lcdb11container_builder.zip)) and follow the [instruction](./experiments/README.md) in [`experiments`](./experiments/) folder. 

### License 
This work is licensed under a CC BY-NC-SA 4.0 - Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### Acknowledgement
This work is primarily based on the [OpenML](https://www.openml.org/) dataset platform and the [Scikit-learn](https://scikit-learn.org/stable/) library.




