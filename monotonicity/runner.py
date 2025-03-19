from py_experimenter.experimenter import PyExperimenter
from experiments import run_experiment_on_X_y
from lcdb import get_dataset, get_schedule
import sys
import pathlib
import json
from tqdm import tqdm
import itertools as it


def run_experiment_outer(keyfields: dict, result_processor, custom_config):
    
    # load dataset
    openmlid = int(keyfields["openmlid"])
    print(f"Loading dataset {openmlid}")
    X, y = get_dataset(openmlid, feature_scaling=False, mix=False, preprocess=False)
    schedule = get_schedule(0.81 * len(X))  # training size is 90% of the 90% not used for test.
    print(schedule)

    # define everything to be covered in this experiment
    domains = {
        "outer_seed": range(5),
        "inner_seed": range(5),
        "anchor": schedule,
        #"feature_scaling": [False, True],
        #"mix": [False, True],
        "realistic": [False, True],
        "fs_realistic": [False, True],
        "algorithm": [
            "lda",
            "SVC_sigmoid",
            "sklearn.neural_network.MLPClassifier",
            "sklearn.linear_model.RidgeClassifier",
            "sklearn.neighbors.NearestCentroid"
            ],
        "monotonicity": ["none", "outer", "inner", "anchor"]
    }
    keys = list(domains.keys())
    all_combinations = list(it.product(*domains.values()))
    
    pbar = tqdm(total=len(all_combinations))
    rows = []
    for combo in all_combinations:
        report = {k: combo[i] for i, k in enumerate(keys)}
        kwargs = report.copy()
        kwargs["X"] = X
        kwargs["y"] = y
        info = run_experiment_on_X_y(**kwargs)
        info["openmlid"] = openmlid
        report.update(info)
        rows.append(report)
        pbar.update(1)
    pbar.close()
    
    result_folder = pathlib.Path("results")
    if not result_folder.exists():
        result_folder.mkdir(parents=True, exist_ok=True)

    with open(f"{result_folder}/{openmlid}.json", "w") as f:
        json.dump(rows, f)

if __name__ == "__main__":
    name = sys.argv[1]

    experimenter = PyExperimenter(
        experiment_configuration_file_path="config.yaml",
        use_codecarbon=False,
        name=name
    )
    #run_experiment_outer(keyfields={"openmlid": 1486}, custom_config=None, result_processor=None)
    #exit(0)
    experimenter.execute(
        experiment_function=run_experiment_outer,
        max_experiments=-1
    )