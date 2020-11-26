from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "Benchmark for `carefree-learn`"
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-learn-benchmark",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=["carefree-learn>=0.1.4"],
    author="carefree0910",
    author_email="syameimaru_kurumi@pku.edu.cn",
    url="https://github.com/carefree0910/carefree-learn-benchmark",
    download_url=f"https://github.com/carefree0910/carefree-learn-benchmark/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python carefree-learn PyTorch",
)
