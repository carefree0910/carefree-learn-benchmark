import os
import cflearn

import numpy as np

from typing import *
from cftool.dist import Parallel
from cftool.misc import hash_code
from cftool.misc import update_dict
from cftool.misc import lock_manager
from cftool.misc import shallow_copy_dict
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from cfdata.tabular import data_type
from cfdata.tabular import task_type_type
from cfdata.tabular import parse_task_type
from cfdata.tabular import KFold
from cfdata.tabular import KRandom
from cfdata.tabular import TabularData
from cfdata.tabular import TabularDataset
from cflearn.dist import Task
from cflearn.dist import Experiments
from cflearn.types import evaluator_type
from cflearn.types import predictor_type
from cftool.ml.utils import Comparer


class BenchmarkResults(NamedTuple):
    data: TabularData
    # external usage
    scores: Optional[np.ndarray]
    # internal usage
    best_configs: Optional[Dict[str, Dict[str, Any]]]
    best_methods: Optional[Dict[str, str]]
    experiments: Optional[Experiments]
    comparer: Optional[Comparer]


class Benchmark(LoggingMixin):
    def __init__(
        self,
        task_name: str,
        task_type: task_type_type,
        *,
        temp_folder: Optional[str] = None,
        project_name: str = "carefree-learn-benchmark",
        models: Union[str, List[str]] = "fcnn",
        increment_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
        read_config: Optional[Dict[str, Any]] = None,
        use_tracker: bool = False,
        use_cuda: bool = True,
    ):
        self.data: Optional[TabularData] = None
        self.experiments: Optional[Experiments] = None

        if data_config is None:
            data_config = {}
        if read_config is None:
            read_config = {}
        self.data_config, self.read_config = data_config, read_config
        self.task_name = task_name
        self.task_type = task_type
        if temp_folder is None:
            temp_folder = f"__{task_name}__"
        self.temp_folder, self.project_name = temp_folder, project_name
        if isinstance(models, str):
            models = [models]
        self.models = models
        if increment_config is None:
            increment_config = {}
        self.increment_config = increment_config
        self.use_tracker = use_tracker
        self.use_cuda = use_cuda

    @property
    def identifier(self) -> str:
        return hash_code(
            f"{self.project_name}{self.task_name}{self.models}{self.increment_config}"
        )

    @property
    def data_tasks(self) -> List[Optional[Task]]:
        experiments = self.experiments
        if experiments is None:
            raise ValueError("`experiments` are not yet generated")
        return next(iter(experiments.data_tasks.values()))

    def _add_tasks(
        self,
        iterator_name: str,
        data_tasks: List[Task],
        experiments: Experiments,
        benchmarks: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> None:
        self.configs: Dict[str, Dict[str, Any]] = {}
        for i in range(len(data_tasks)):
            for model in self.models:
                model_benchmarks = benchmarks.get(model)
                if model_benchmarks is None:
                    model_benchmarks = cflearn.Zoo(model).benchmarks
                for model_type, config in model_benchmarks.items():
                    identifier = f"{model}_{self.task_name}_{model_type}"
                    task_name = f"{identifier}_{iterator_name}{i}"
                    increment_config = shallow_copy_dict(self.increment_config)
                    config = update_dict(increment_config, config)
                    self.configs.setdefault(identifier, config)
                    if not self.use_tracker:
                        tracker_config = None
                    else:
                        tracker_config = {
                            "project_name": self.project_name,
                            "task_name": task_name,
                            "overwrite": True,
                        }
                    experiments.add_task(
                        model=model,
                        data_task=data_tasks[i],
                        identifier=identifier,
                        tracker_config=tracker_config,
                        **config,
                    )

    def _run_tasks(
        self,
        num_jobs: int = 4,
        run_tasks: bool = True,
        predict_config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResults:
        experiments = self.experiments
        if experiments is None:
            raise ValueError("`experiments` is not yet defined")
        results: Dict[str, List[Any]] = experiments.run_tasks(
            num_jobs=num_jobs,
            run_tasks=run_tasks,
            load_task=cflearn.load_task,
        )
        comparer_list = []
        for i, data_task in enumerate(self.data_tasks):
            assert data_task is not None
            pipelines = {}
            x_te, y_te = data_task.fetch_data("_te")
            for identifier, ms in results.items():
                pipelines[identifier] = ms[i]
            comparer = cflearn.evaluate(
                x_te,
                y_te,
                pipelines=pipelines,
                predict_config=predict_config,
                comparer_verbose_level=None,
            )
            comparer_list.append(comparer)
        comparer = Comparer.merge(comparer_list)
        best_methods = comparer.best_methods
        best_configs = {
            metric: self.configs[identifier]
            for metric, identifier in best_methods.items()
        }
        return BenchmarkResults(
            self.data,
            None,
            best_configs,
            best_methods,
            experiments,
            comparer,
        )

    def _pre_process(self, x: data_type, y: data_type = None) -> TabularDataset:
        data_config = shallow_copy_dict(self.data_config)
        task_type = data_config.pop("task_type", None)
        if task_type is not None:
            assert parse_task_type(task_type) is parse_task_type(self.task_type)
        self.data = TabularData.simple(self.task_type, **data_config)
        self.data.read(x, y, **self.read_config)
        return self.data.to_dataset()

    def _k_core(
        self,
        k_iterator: Iterable,
        num_jobs: int,
        predictor: predictor_type,
        evaluator: evaluator_type,
        run_tasks: bool,
        predict_config: Optional[Dict[str, Any]],
        benchmarks: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    ) -> BenchmarkResults:
        if predictor is not None and evaluator is not None:
            indices = []
            train_datasets = []
            test_features, test_labels = [], []
            for i, (train_split, test_split) in enumerate(k_iterator):
                train_datasets.append(train_split.dataset)
                te_x, te_y = test_split.dataset.xy
                test_features.append(te_x)
                test_labels.append(te_y)
                indices.append(i)
            parallel = Parallel(num_jobs)
            parallel = parallel(predictor, indices, train_datasets, test_features)
            predictions_list = parallel.ordered_results
            return BenchmarkResults(
                self.data,
                evaluator(predictions_list, test_labels),
                None,
                None,
                None,
                None,
            )
        if benchmarks is None:
            benchmarks = {}
        self.experiments = Experiments(self.temp_folder, use_cuda=self.use_cuda)
        data_tasks = []
        for i, (train_split, test_split) in enumerate(k_iterator):
            train_dataset, test_dataset = train_split.dataset, test_split.dataset
            x_tr, y_tr = train_dataset.xy
            x_te, y_te = test_dataset.xy
            data_task = Task.data_task(i, self.identifier, self.experiments)
            data_task.dump_data(x_tr, y_tr)
            data_task.dump_data(x_te, y_te, "_te")
            data_tasks.append(data_task)
        self._iterator_name = type(k_iterator).__name__
        self._add_tasks(self._iterator_name, data_tasks, self.experiments, benchmarks)
        return self._run_tasks(num_jobs, run_tasks, predict_config)

    def k_fold(
        self,
        k: int,
        x: data_type,
        y: data_type = None,
        *,
        num_jobs: int = 4,
        # external usage
        predictor: predictor_type = None,
        evaluator: evaluator_type = None,
        # internal usage
        run_tasks: bool = True,
        predict_config: Optional[Dict[str, Any]] = None,
        benchmarks: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ) -> BenchmarkResults:
        dataset = self._pre_process(x, y)
        return self._k_core(
            KFold(k, dataset),
            num_jobs,
            predictor,
            evaluator,
            run_tasks,
            predict_config,
            benchmarks,
        )

    def k_random(
        self,
        k: int,
        num_test: Union[int, float],
        x: data_type,
        y: data_type = None,
        *,
        num_jobs: int = 4,
        # external usage
        predictor: predictor_type = None,
        evaluator: evaluator_type = None,
        # internal usage
        run_tasks: bool = True,
        predict_config: Optional[Dict[str, Any]] = None,
        benchmarks: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ) -> BenchmarkResults:
        dataset = self._pre_process(x, y)
        return self._k_core(
            KRandom(k, num_test, dataset),
            num_jobs,
            predictor,
            evaluator,
            run_tasks,
            predict_config,
            benchmarks,
        )

    def save(
        self,
        export_folder: str,
        *,
        simplify: bool = True,
        compress: bool = True,
    ) -> "Benchmark":
        experiments = self.experiments
        if experiments is None:
            raise ValueError("`experiments` is not yet defined")
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            Saving.prepare_folder(self, export_folder)
            if isinstance(self.task_type, str):
                task_type_value = self.task_type
            else:
                task_type_value = self.task_type.value
            Saving.save_dict(
                {
                    "task_name": self.task_name,
                    "task_type": task_type_value,
                    "project_name": self.project_name,
                    "models": self.models,
                    "increment_config": self.increment_config,
                    "use_cuda": self.use_cuda,
                    "iterator_name": self._iterator_name,
                    "temp_folder": self.temp_folder,
                    "configs": self.configs,
                },
                "kwargs",
                abs_folder,
            )
            experiments_folder = os.path.join(abs_folder, "__experiments__")
            experiments.save(
                experiments_folder,
                simplify=simplify,
                compress=compress,
            )
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self

    @classmethod
    def load(
        cls,
        saving_folder: str,
        *,
        predict_config: Optional[Dict[str, Any]] = None,
        compress: bool = True,
    ) -> Tuple["Benchmark", BenchmarkResults]:
        abs_folder = os.path.abspath(saving_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [saving_folder]):
            with Saving.compress_loader(abs_folder, compress, remove_extracted=False):
                kwargs = Saving.load_dict("kwargs", abs_folder)
                configs = kwargs.pop("configs")
                iterator_name = kwargs.pop("iterator_name")
                benchmark = cls(**shallow_copy_dict(kwargs))
                benchmark.configs = configs
                benchmark._iterator_name = iterator_name
                benchmark.experiments = Experiments.load(
                    os.path.join(abs_folder, "__experiments__")
                )
                results = benchmark._run_tasks(0, False, predict_config)
        return benchmark, results


__all__ = ["BenchmarkResults", "Benchmark"]
