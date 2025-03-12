from unittest import TestCase
from parameterized import parameterized
import numpy as np


from lcdb import get_dataset, get_splits_for_anchor, get_truth_and_predictions
from sklearn.svm import LinearSVC


class TestLCDB(TestCase):

    @parameterized.expand([
        (61)
    ])
    def test_dataset_reproducibility(self, openmlid):
    
        # load dataset (and check that it is always the same)
        data = []
        for _ in range(2):
            data.append(get_dataset(openmlid=openmlid, feature_scaling=True, mix=True))
        self.assertEqual(str(data[0]), str(data[1]))
        X, y = data[0]

    @parameterized.expand([
        (61)
    ])
    def test_split_reproducibility(self, openmlid):

        X, y = get_dataset(openmlid=openmlid, feature_scaling=True, mix=True)

        # get_splits_for_anchor
        for anchor in [16, 32, 64]:
            for monotonic in [False, True]:
                for seed in range(5):
                    splits = []
                    for _ in range(2):
                        splits.append(get_splits_for_anchor(
                            X=X,
                            y=y,
                            anchor=anchor,
                            outer_seed=seed,
                            inner_seed=seed,
                            monotonic=monotonic
                            ))
                    self.assertEqual(str(splits[0]), str(splits[1]))
                    
    @parameterized.expand([
        (61)
    ])
    def test_split_sensibility_to_outer_seeds(self, openmlid):

        X, y = get_dataset(openmlid=openmlid, feature_scaling=True, mix=True)

        # get_splits_for_anchor
        for anchor in [16, 32, 64]:
            for monotonic in [False, True]:
                for seed in range(5):
                    splits = []
                    for _ in range(1, 3):
                        splits.append(get_splits_for_anchor(
                            X=X,
                            y=y,
                            anchor=anchor,
                            outer_seed=seed + _,
                            inner_seed=seed,
                            monotonic=monotonic
                            ))
                    self.assertNotEqual(str(splits[0]), str(splits[1]))

    @parameterized.expand([
        (61)
    ])
    def test_split_sensibility_to_inner_seeds(self, openmlid):

        X, y = get_dataset(openmlid=openmlid, feature_scaling=True, mix=True)

        # get_splits_for_anchor
        for anchor in [16, 32, 64]:
            for monotonic in [False, True]:
                for seed in range(5):
                    splits = []
                    for _ in range(1, 3):
                        splits.append(get_splits_for_anchor(
                            X=X,
                            y=y,
                            anchor=anchor,
                            outer_seed=seed,
                            inner_seed=seed + _,
                            monotonic=monotonic
                            ))
                    self.assertNotEqual(str(splits[0]), str(splits[1]))
    
    @parameterized.expand([
        (61)
    ])
    def test_training_set_size_at_anchors(self, openmlid):
        X, y = get_dataset(openmlid=openmlid, feature_scaling=True, mix=True)

        # get_splits_for_anchor
        for anchor in [16, 32, 64]:
            for monotonic in [False, True]:
                for seed in range(5):
                    X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(
                        X=X,
                        y=y,
                        anchor=anchor,
                        outer_seed=seed,
                        inner_seed=seed,
                        monotonic=monotonic
                        )
                    self.assertEqual(anchor, len(X_train))

    @parameterized.expand([
        (60)
    ])
    def test_sensitivity_to_monotonicity(self, openmlid):
        X, y = get_dataset(openmlid=openmlid, feature_scaling=True, mix=True)

        # get_splits_for_anchor
        for monotonic in [False, True]:
            for seed in range(5):
                training_sets = []
                for anchor in [16, 32, 64]:
                    X_train, _, _, _, _, _ = get_splits_for_anchor(
                        X=X,
                        y=y,
                        anchor=anchor,
                        outer_seed=seed,
                        inner_seed=seed,
                        monotonic=monotonic
                        )
                    
                    if training_sets:
                        self.assertEqual(training_sets[-1].shape, X_train[:len(training_sets[-1])].shape)
                        if monotonic:
                            self.assertTrue(np.array_equal(training_sets[-1], X_train[:len(training_sets[-1])]))
                        else:
                            self.assertFalse(np.array_equal(training_sets[-1], X_train[:len(training_sets[-1])]))

                    training_sets.append(X_train)
    

    @parameterized.expand([
        (61, "sklearn.svm.LinearSVC", {"random_state": 0})
    ])
    def test_reproducibility_of_entry(self, openmlid, learner_name, learner_params):
        X, y = get_dataset(openmlid=openmlid, mix=False, feature_scaling=True)

        for anchor in [16]:
            for realistic in [False]:#, True]:
                for fs_realistic in [False, True]:
                    for monotonic in [False, True]:

                        entries = []
                        for _ in range(2):
                            learner_inst = LinearSVC(random_state=0)
                            entries.append(get_truth_and_predictions(
                                learner_inst,
                                X,
                                y,
                                anchor,
                                outer_seed=0,
                                inner_seed=0,
                                realistic=realistic,
                                fs_realisic=fs_realistic,
                                monotonic=monotonic
                            ))
                        for name, v1, v2 in zip(range(8), entries[0], entries[1]):
                            self.assertEqual(str(v1), str(v2), f"Inconsistency in field {name}. First is {v1} but second {v2}")
