# LCDB 1.1: A Database Illustrating Learning Curves are more Ill-Behaved than Previously Thought

<p align="center">
  <a href="https://arxiv.org/abs/2505.15657" target="_blank"><img src="https://img.shields.io/badge/arXiv-2505.15657-B31B1B"></a>
  <a href="https://data.4tu.nl/private_datasets/V7dDlGyQJqPc_mXUAJL1MweACKG557GQtOWIVHhYpjU" target="_blank"><img src="https://img.shields.io/badge/4TU.ResearchData-LCDB 1.1-orange"></a>
  <a href="https://neurips.cc/virtual/2025/poster/121511" target="_blank">
    <img src="https://img.shields.io/badge/NeurIPS%20-2025-7A1FA2?logo=neurips&logoColor=white">
  </a>
</p>

### Project Structure
```
📦 Project Root
├── 📂 analysis           — Analysis notebooks
├── 📂 dataset            — The LCDB 1.1 datasets 
├── 📂 experiments        — Files for organising the large-scale experiments
├── 📂 lcdb_function      — Functions for computing the learning curves
└── LCDB11_demo.ipynb     - showing how to use our database
```

### Updated LCDB 1.1 Features 
| Database   | Data Acquisition             | Feature Scaling | Estimation Anchor               | Learner \| Dataset  |
|------------|--------------------------|-----------------|---------------------------------|----------------------|
| **LCDB 1.0**  | with data-leakage (dl)    | none              | ⌈ 16 ⋅ 2<sup>n/2</sup> ⌉               | 20 \| 196  |
| **LCDB 1.1**  | with and without dl   | none, min-max, standard        | 4 times denser                 | 32 \| 265    |

### How to use the LCDB 1.1? 
First download the precomputed learning curves from [4TU.ResearchData](https://data.4tu.nl/private_datasets/V7dDlGyQJqPc_mXUAJL1MweACKG557GQtOWIVHhYpjU) and extract them to the folder [`dataset`](./dataset/). The [Readme](./dataset/README.md) in the dataset folder explains the files and their content. Or you can use the demonstration in [`LCDB11_demo.ipynb`](./LCDB11_demo.ipynb) to download the data automatically. Useful metafeatures can be found in folder [`meta_feature`](./meta_feature/). 

Note, we recommend using learning curves from validation sets, since the sets differ in each inner and outer split, to ensure a no data-leakage version. The demostration can be found in [`LCDB11_demo.ipynb`](./LCDB11_demo.ipynb). 

### Reproduce all figures from the paper
Install the packages in [`analysis/requirements.txt`](./analysis/requirements.txt). Once the setup is complete,the analysis notebooks in the [`analysis`](./analysis/) folder illustrate the usage. 

### Contact us if you need more metrics!
We have stored all the probabilistic outputs and / or scores (when available for the learner). However, these files are too large for us to host for the broader public. If you are interested in a particular metric that we do not include, please contact us and we can compute it and host it. 

### Workflow for Computing LCDB 1.1
To ensure the reproducibility, please download the [container](https://surfdrive.surf.nl/files/index.php/s/TSe0nqWKcT5jPwK) (or generate it by using  or use the [`env`](./experiments/env) to generate the same image) and follow the [instruction](./experiments/README.md) in [`experiments`](./experiments/) folder. 

### License 
This work is licensed under a CC BY 4.0 - - Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/). 

### Acknowledgement
This work is primarily based on the [OpenML](https://www.openml.org/) dataset platform and the [Scikit-learn](https://scikit-learn.org/stable/) library.




