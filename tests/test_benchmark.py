import os
import cflearn
import platform

from cfdata.tabular import TabularDataset
from cflearn_benchmark import Benchmark

num_jobs = 0 if platform.system() == "Linux" else 2
logging_folder = "__test_dist__"
kwargs = {"fixed_epoch": 3}


def test_benchmark() -> None:
    benchmark_folder = os.path.join(logging_folder, "__test_benchmark__")
    x, y = TabularDataset.iris().xy
    benchmark = Benchmark(
        "foo",
        "clf",
        models=["fcnn", "tree_dnn"],
        temp_folder=benchmark_folder,
        increment_config=kwargs.copy(),
    )
    benchmarks = {
        "fcnn": {"default": {}, "sgd": {"optimizer": "sgd"}},
        "tree_dnn": {"default": {}, "adamw": {"optimizer": "adamw"}},
    }
    results = benchmark.k_fold(
        3,
        x,
        y,
        num_jobs=num_jobs,
        benchmarks=benchmarks,  # type: ignore
    )
    msg1 = results.comparer.log_statistics()
    saving_folder = os.path.join(logging_folder, "__test_benchmark_save__")
    benchmark.save(saving_folder)
    loaded_benchmark, loaded_results = Benchmark.load(saving_folder)
    msg2 = loaded_results.comparer.log_statistics()
    assert msg1 == msg2
    cflearn._rmtree(logging_folder)


if __name__ == "__main__":
    test_benchmark()
