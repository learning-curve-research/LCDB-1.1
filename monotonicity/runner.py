from py_experimenter.experimenter import PyExperimenter
from experiments import run_experiment_on_X_y
from lcdb import get_dataset, get_schedule
import sys
import pathlib
import json
from tqdm import tqdm


def run_experiment_outer(keyfields: dict, result_processor, custom_config):
    
    # load dataset
    openmlid = int(keyfields["openmlid"])
    feature_scaling = bool(keyfields["feature_scaling"])
    mix = bool(keyfields["mix"])
    realistic = bool(keyfields["realistic"])
    print(f"Loading dataset {openmlid}")
    X, y = get_dataset(openmlid, feature_scaling=feature_scaling, mix=mix, preprocess=not realistic)
    schedule = get_schedule(0.81 * len(X))  # training size is 90% of the 90% not used for test.
    print(schedule)

    outer_seeds = range(5)
    inner_seeds = range(5)

    pbar = tqdm(total=len(outer_seeds) * len(inner_seeds) * len(schedule))
    report = {}
    for outer_seed in outer_seeds:
        report[str(outer_seed)] = {}
        for inner_seed in inner_seeds:
            report[str(outer_seed)][str(inner_seed)] = {}
            for a in schedule:
                report[str(outer_seed)][str(inner_seed)][str(a)] = run_experiment_on_X_y(
                    X=X,
                    y=y,
                    realistic=realistic,
                    algorithm=keyfields["algorithm"],
                    outer_seed=outer_seed,
                    inner_seed=inner_seed,
                    anchor=a,
                    fs_realistic=keyfields["fs_realistic"],
                    monotonic=keyfields["monotonic"]
                )
                pbar.update(1)
    pbar.close()
    
    result_folder = pathlib.Path("results")
    if not result_folder.exists():
        result_folder.mkdir(parents=True, exist_ok=True)

    with open(f"{result_folder}/{keyfields}.json", "w") as f:
        json.dump(report, f)

if __name__ == "__main__":
    name = sys.argv[1]

    experimenter = PyExperimenter(
        experiment_configuration_file_path="config.yaml",
        use_codecarbon=False,
        name=name
    )
    experimenter.execute(
        experiment_function=run_experiment_outer,
        max_experiments=-1
        )