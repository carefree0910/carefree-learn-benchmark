import os
import shutil
import cflearn

from typing import *
from cftool.misc import timestamp
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
from cflearn.dist import Experiment
from cflearn.pipeline import Pipeline
from cftool.ml.utils import Comparer


class BenchmarkResults(NamedTuple):
    best_configs: Optional[Dict[str, Dict[str, Any]]]
    best_methods: Optional[Dict[str, str]]
    comparer: Optional[Comparer]


class Benchmark:
    data_name = "__data__"
    experiment_name = "__experiment__"

    def __init__(
        self,
        task_name: str,
        task_type: task_type_type,
        *,
        use_mlflow: bool = True,
        temp_folder: Optional[str] = None,
        models: Union[str, List[str]] = "fcnn",
        data_config: Optional[Dict[str, Any]] = None,
        read_config: Optional[Dict[str, Any]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
        benchmarks: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        num_jobs: int = 4,
        use_cuda: bool = True,
        available_cuda_list: Optional[List[int]] = None,
        is_loading: bool = False,
    ):
        self.data: Optional[TabularData] = None
        self.results: Optional[BenchmarkResults] = None
        self.predict_config: Optional[Dict[str, Any]] = None

        self.use_mlflow = use_mlflow
        self.experiment = Experiment(
            num_jobs=num_jobs,
            use_cuda=use_cuda,
            available_cuda_list=available_cuda_list,
        )

        self.task_name = task_name
        self.task_type = task_type
        self.mlflow_task_name = f"{task_name}_{timestamp()}"

        self.data_config = data_config or {}
        self.read_config = read_config or {}
        self.increment_config = increment_config or {}
        self.benchmarks = benchmarks or {}

        if temp_folder is None:
            temp_folder = f"__{task_name}__"
        if not is_loading:
            if os.path.isdir(temp_folder):
                print(
                    f"{LoggingMixin.warning_prefix}'{temp_folder}' already exists, "
                    "it will be erased to store our logging"
                )
                shutil.rmtree(temp_folder)
            os.makedirs(temp_folder)
        self.temp_folder = temp_folder

        if isinstance(models, str):
            models = [models]
        self.models = models

    def _add_tasks(self, iterator_name: str, data_folders: List[str]) -> None:
        self.data_folders = data_folders
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.data_folder2workplaces: Dict[str, List[str]] = {}
        self.workplace2model_setting: Dict[str, str] = {}
        for data_folder in data_folders:
            local_workplaces = []
            for model in self.models:
                model_benchmarks = self.benchmarks.get(model)
                if model_benchmarks is None:
                    model_benchmarks = cflearn.Zoo(model).benchmarks
                for model_type, config in model_benchmarks.items():
                    model_setting = f"{model}_{model_type}"
                    if not self.use_mlflow:
                        mlflow_config = None
                    else:
                        mlflow_params = shallow_copy_dict(config)
                        mlflow_params["model"] = model
                        mlflow_config = {
                            "task_name": self.mlflow_task_name,
                            "run_name": f"{model_setting}_{iterator_name}",
                            "mlflow_params": mlflow_params,
                        }
                    increment_config = shallow_copy_dict(self.increment_config)
                    increment_config["mlflow_config"] = mlflow_config
                    kwargs = update_dict(increment_config, shallow_copy_dict(config))
                    self.configs.setdefault(model_setting, kwargs)
                    workplace = self.experiment.add_task(
                        model=model,
                        data_folder=data_folder,
                        root_workplace=self.temp_folder,
                        config=kwargs,
                    )
                    self.workplace2model_setting[workplace] = model_setting
                    local_workplaces.append(workplace)
            self.data_folder2workplaces[data_folder] = local_workplaces

    def _merge_comparer(self, pipeline_dict: Dict[str, Pipeline]) -> Comparer:
        comparer_list = []
        for data_folder in self.data_folders:
            local_workplaces = self.data_folder2workplaces[data_folder]
            local_pipelines = {
                self.workplace2model_setting[workplace]: pipeline_dict[workplace]
                for workplace in local_workplaces
            }
            x_te, y_te = self.experiment.fetch_data("_te", data_folder=data_folder)
            comparer = cflearn.evaluate(
                x_te,
                y_te,
                pipelines=local_pipelines,
                predict_config=self.predict_config,
                comparer_verbose_level=None,
            )
            comparer_list.append(comparer)
        return Comparer.merge(comparer_list)

    def _run_tasks(self, load_tasks: bool) -> BenchmarkResults:
        experiment = self.experiment
        if experiment is None:
            raise ValueError("`experiment` is not yet defined")
        task_loader = cflearn.task_loader if load_tasks else None
        results = experiment.run_tasks(task_loader=task_loader)
        pipeline_dict = results.pipeline_dict
        comparer = self._merge_comparer(pipeline_dict)
        best_methods = comparer.best_methods
        best_configs = {
            metric: self.configs[identifier]
            for metric, identifier in best_methods.items()
        }
        return BenchmarkResults(best_configs, best_methods, comparer)

    def _pre_process(self, x: data_type, y: data_type = None) -> TabularDataset:
        data_config = shallow_copy_dict(self.data_config)
        task_type = data_config.pop("task_type", None)
        if task_type is not None:
            assert parse_task_type(task_type) is parse_task_type(self.task_type)
        self.data = TabularData.simple(self.task_type, **data_config)
        self.data.read(x, y, **self.read_config)
        return self.data.to_dataset()

    def _data_folder(self, i: int) -> str:
        folder = os.path.join(self.temp_folder, "__data__", str(i))
        os.makedirs(folder, exist_ok=True)
        return folder

    def _k_core(
        self,
        k_iterator: Iterable,
        load_tasks: bool,
        predict_config: Optional[Dict[str, Any]],
    ) -> BenchmarkResults:
        data_folders = []
        for i, (train_split, test_split) in enumerate(k_iterator):
            train_dataset, test_dataset = train_split.dataset, test_split.dataset
            x_tr, y_tr = train_dataset.xy
            x_te, y_te = test_dataset.xy
            data_folder = self._data_folder(i)
            self.experiment.dump_data(data_folder, x_tr, y_tr)
            self.experiment.dump_data(data_folder, x_te, y_te, "_te")
            data_folders.append(data_folder)
        self._add_tasks(type(k_iterator).__name__, data_folders)
        self.predict_config = predict_config
        self.results = self._run_tasks(load_tasks)
        return self.results

    def k_fold(
        self,
        k: int,
        x: data_type,
        y: data_type = None,
        *,
        load_tasks: bool = True,
        predict_config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResults:
        dataset = self._pre_process(x, y)
        return self._k_core(KFold(k, dataset), load_tasks, predict_config)

    def k_random(
        self,
        k: int,
        num_test: Union[int, float],
        x: data_type,
        y: data_type = None,
        *,
        load_tasks: bool = True,
        predict_config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResults:
        dataset = self._pre_process(x, y)
        return self._k_core(KRandom(k, num_test, dataset), load_tasks, predict_config)

    def save(self, export_folder: str, *, compress: bool = True) -> "Benchmark":
        if self.data is None:
            raise ValueError("`data` is not yet defined")
        if self.experiment is None:
            raise ValueError("`experiment` is not yet defined")
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            Saving.prepare_folder(self, export_folder)
            if isinstance(self.task_type, str):
                task_type_value = self.task_type
            else:
                task_type_value = self.task_type.value
            # configs
            saving_results = self.results
            if saving_results is not None:
                saving_results = BenchmarkResults(
                    saving_results.best_configs,
                    saving_results.best_methods,
                    None,
                )
            Saving.save_dict(
                {
                    "task_name": self.task_name,
                    "task_type": task_type_value,
                    "models": self.models,
                    "increment_config": self.increment_config,
                    "temp_folder": self.temp_folder,
                    "data_folders": self.data_folders,
                    "configs": self.configs,
                    "predict_config": self.predict_config,
                    "results": saving_results,
                    "mlflow_task_name": self.mlflow_task_name,
                    "data_folder2workplaces": self.data_folder2workplaces,
                    "workplace2model_setting": self.workplace2model_setting,
                },
                "kwargs",
                abs_folder,
            )
            # data
            data_folder = os.path.join(abs_folder, self.data_name)
            self.data.save(data_folder, compress=False)
            # experiment
            experiment_folder = os.path.join(abs_folder, self.experiment_name)
            self.experiment.save(experiment_folder, compress=False)
            # compress
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self

    @classmethod
    def load(
        cls,
        saving_folder: str,
        *,
        compress: bool = True,
        load_comparer: bool = True,
    ) -> "Benchmark":
        abs_folder = os.path.abspath(saving_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [saving_folder]):
            with Saving.compress_loader(abs_folder, compress, remove_extracted=False):
                # configs
                kwargs = Saving.load_dict("kwargs", abs_folder)
                configs = kwargs.pop("configs")
                predict_config = kwargs.pop("predict_config")
                results = list(kwargs.pop("results"))
                mlflow_task_name = kwargs.pop("mlflow_task_name")
                data_folders = kwargs.pop("data_folders")
                data_folder2workplaces = kwargs.pop("data_folder2workplaces")
                workplace2model_setting = kwargs.pop("workplace2model_setting")
                benchmark = cls(is_loading=True, **shallow_copy_dict(kwargs))
                benchmark.configs = configs
                benchmark.predict_config = predict_config
                benchmark.mlflow_task_name = mlflow_task_name
                benchmark.data_folders = data_folders
                benchmark.data_folder2workplaces = data_folder2workplaces
                benchmark.workplace2model_setting = workplace2model_setting
                # data
                data_folder = os.path.join(abs_folder, cls.data_name)
                data = TabularData.load(data_folder, compress=False)
                # experiment
                experiment_folder = os.path.join(abs_folder, cls.experiment_name)
                experiment = Experiment.load(
                    experiment_folder,
                    compress=False,
                    task_loader=cflearn.task_loader if load_comparer else None,
                )
                # results
                comparer = None
                if load_comparer:
                    pipeline_dict = experiment.results.pipeline_dict
                    comparer = benchmark._merge_comparer(pipeline_dict)
                results[-1] = comparer
                benchmark.results = BenchmarkResults(*results)
                # assign
                benchmark.data = data
                benchmark.experiment = experiment
        return benchmark


__all__ = ["BenchmarkResults", "Benchmark"]
