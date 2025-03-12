import numpy as np

from lcdb import get_dataset, get_entry_learner
import sys, traceback

def run_experiment_on_X_y(X, y,  realistic, fs_realistic, algorithm, outer_seed, inner_seed, anchor, monotonic):

    info = {
            "realistic": realistic,
            "fs_realistic": fs_realistic,
            "learner": algorithm,
            "outer_seed": outer_seed,
            "inner_seed": inner_seed,
            "anchor": anchor
        }

    # seeding should be done just before computing the learning curve
    SEED = 42
    np.random.seed(SEED)

    # Mapping of algorithms to their corresponding parameters
    defined_params = {
        "SVC_linear": ("sklearn.svm.LinearSVC", {"random_state": SEED}),
        "SVC_poly": ("sklearn.svm.SVC", {"kernel": "poly", "random_state": SEED}),
        "SVC_rbf": ("sklearn.svm.SVC", {"kernel": "rbf", "random_state": SEED}),
        "SVC_sigmoid": ("sklearn.svm.SVC", {"kernel": "sigmoid", "random_state": SEED}),
        "lda": ("sklearn.discriminant_analysis.LinearDiscriminantAnalysis", {"random_state": SEED}),
        "sklearn.tree.DecisionTreeClassifier": ("sklearn.tree.DecisionTreeClassifier", {"random_state": SEED}),
        "sklearn.tree.ExtraTreeClassifier": ("sklearn.tree.ExtraTreeClassifier", {"random_state": SEED}),
        "sklearn.linear_model.LogisticRegression": ("sklearn.linear_model.LogisticRegression", {"random_state": SEED}),
        "sklearn.linear_model.PassiveAggressiveClassifier": ("sklearn.linear_model.PassiveAggressiveClassifier", {"random_state": SEED}),
        "sklearn.linear_model.Perceptron": ("sklearn.linear_model.Perceptron", {"random_state": SEED}),
        "sklearn.linear_model.RidgeClassifier": ("sklearn.linear_model.RidgeClassifier", {"random_state": SEED}),
        "sklearn.linear_model.SGDClassifier": ("sklearn.linear_model.SGDClassifier", {"random_state": SEED}),
        "sklearn.neural_network.MLPClassifier": ("sklearn.neural_network.MLPClassifier", {"random_state": SEED}),
        "sklearn.ensemble.ExtraTreesClassifier": ("sklearn.ensemble.ExtraTreesClassifier", {"random_state": SEED}),
        "sklearn.ensemble.RandomForestClassifier": ("sklearn.ensemble.RandomForestClassifier", {"random_state": SEED}),
        "sklearn.ensemble.GradientBoostingClassifier": ("sklearn.ensemble.GradientBoostingClassifier", {"random_state": SEED}),
        "sklearn.dummy.DummyClassifier": ("sklearn.dummy.DummyClassifier", {"strategy": "most_frequent"})
    }

    # Get algorithm and parameters
    if algorithm in defined_params:
        learner_name, learner_params = defined_params[algorithm]
    else:
        learner_name = algorithm
        learner_params = {}

    status = "init"

    # the default context sometimes runs into errors on the cluster
    # the error is: `[Errno 24] Too many open files`
    # actually, this error may be caused by something else... its resolved via 'ulimit -n 8000'

    try:
        info.update(
            get_entry_learner(
                learner_name=learner_name,
                learner_params=learner_params,
                X=X,
                y=y,
                anchor=anchor,
                outer_seed=outer_seed,
                inner_seed=inner_seed,
                realistic=realistic,
                fs_realisic=fs_realistic,
                monotonic=monotonic)
        )
        status = "ok"
    
    except Exception as e:
        print(f"AN ERROR OCCURED! Here are the details.\n{type(e)}\n{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(traceback_string)

        info.update({
            "error_type": str(type(e)),
            "error_message": str(e),
            "error_traceback": traceback_string,
        })
        status = "error"

    info["status"] = status

    return info


def abc():
    print(f"Loading dataset {openmlid}")
    X, y = get_dataset(openmlid, feature_scaling=feature_scaling, mix=mix, preprocess=~realistic)
    labels = list(np.unique(y))
    is_binary = len(labels) == 2
    minority_class = labels[np.argmin([np.count_nonzero(y == label) for label in labels])]
    print(f"Labels are: {labels}")
    print(f"minority_class is {minority_class}")
    print("Now computing the anchor ")