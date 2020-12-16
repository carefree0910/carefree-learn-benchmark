import os
import cflearn
import platform
import unittest

import numpy as np

from typing import Dict
from cflearn_benchmark import Benchmark
from scipy.sparse import csr_matrix
from cftool.ml import patterns_type
from cftool.ml import Tracker
from cftool.ml import Comparer
from cftool.misc import timestamp
from cfdata.tabular import TabularData
from cfml.misc.toolkit import Experiment
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class TestOpenML(unittest.TestCase):
    Experiment.suppress_warnings()

    num_repeat = 3
    num_jobs = 0 if platform.system() == "Linux" else 2
    project_name = "carefree-learn-benchmark"
    logging_folder = "__test_openml__"
    openml_indices = [38, 389]
    # openml_indices = [38, 46, 179, 184, 389, 554, 772, 917, 1049, 1111, 1120, 1128, 293]
    task_names = [f"openml_{openml_id}" for openml_id in openml_indices]
    messages: Dict[str, str] = {}

    def _get_benchmark_saving_folder(self, task_name: str) -> str:
        benchmark_sub_folder = os.path.join("benchmarks", f"{task_name}_benchmark")
        return os.path.join(self.logging_folder, benchmark_sub_folder)

    def test1(self) -> None:
        for openml_id, task_name in zip(self.openml_indices, self.task_names):
            # preparation
            bunch = fetch_openml(data_id=openml_id)
            x, y = bunch.data, bunch.target
            if isinstance(x, csr_matrix):
                x = x.toarray()
            feature_names = bunch.feature_names
            if bunch.categories is None:
                categorical_columns = None
            else:
                categorical_columns = [
                    i
                    for i, name in enumerate(feature_names)
                    if name in bunch.categories
                ]
            data = TabularData(
                process_methods=None,
                valid_columns=list(range(x.shape[1])),
                categorical_columns=categorical_columns,
            )
            data.read(x, y.reshape([-1, 1]))

            comparer_list = []
            sk_bases = [
                LinearSVC,
                SVC,
                DecisionTreeClassifier,
                RandomForestClassifier,
                LogisticRegression,
            ]

            # cflearn benchmark
            benchmark = Benchmark(
                task_name,
                "clf",
                num_jobs=self.num_jobs,
                models=["fcnn", "tree_dnn"],
                temp_folder=os.path.join(
                    self.logging_folder,
                    f"__test_openml_{openml_id}__",
                ),
                increment_config={
                    "fixed_epoch": 2,
                    "data_config": {"categorical_columns": categorical_columns},
                },
            )
            results = benchmark.k_random(self.num_repeat, 0.1, *data.converted.xy)
            msg = results.comparer.log_statistics(verbose_level=None)
            TestOpenML.messages[task_name] = msg
            benchmark.save(self._get_benchmark_saving_folder(task_name))
            best_methods = list(set(results.best_methods.values()))
            comparer_list.append(results.comparer.select(best_methods))

            # sklearn
            exp = benchmark.experiment
            data_folders = benchmark.data_folders
            for data_folder in data_folders:
                sklearn_patterns: Dict[str, patterns_type] = {}
                x_tr, y_tr = exp.fetch_data(data_folder=data_folder)
                x_te, y_te = exp.fetch_data("_te", data_folder=data_folder)
                assert isinstance(y_tr, np.ndarray)
                for base in sk_bases:
                    clf = base()
                    sklearn_patterns.setdefault(base.__name__, []).append(
                        cflearn.ModelPattern(
                            init_method=lambda: clf.fit(x_tr, y_tr.ravel()),
                            predict_method=lambda x_: clf.predict(x_).reshape([-1, 1]),
                            predict_prob_method="predict_proba",
                        )
                    )
                comparer_list.append(
                    cflearn.evaluate(
                        x_te,
                        y_te,
                        metrics=["acc", "auc"],
                        other_patterns=sklearn_patterns,
                        comparer_verbose_level=None,
                    )
                )

            comparer = Comparer.merge(comparer_list)
            msg = comparer.log_statistics(method_length=24)
            tracker = Tracker(self.project_name, f"{task_name}_summary")
            tracker.track_message(timestamp(), msg)

    def test2(self) -> None:
        for task_name in self.task_names:
            saving_folder = self._get_benchmark_saving_folder(task_name)
            benchmark = Benchmark.load(saving_folder)
            loaded_msg = benchmark.results.comparer.log_statistics(verbose_level=None)
            self.assertEqual(TestOpenML.messages[task_name], loaded_msg)
        cflearn._rmtree(self.logging_folder)


if __name__ == "__main__":
    unittest.main()
