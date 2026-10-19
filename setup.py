from pathlib import Path

from setuptools import find_packages, setup

_NODEPS_PKGS = {
    "momentfm",
    "uni2ts",
    "tirex-2",
    "xlstm",
    "flashrnn",
    "mlstm-kernels",
}

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "requirements.txt") as file:
    required_packages = [
        ln.strip()
        for ln in file
        if ln.strip()
        and not ln.startswith("#")
        and not any(ln.strip().startswith(p) for p in _NODEPS_PKGS)
    ]

style_packages = [
    "black==23.3.0",
    "flake8==6.0.0",
    "mypy==1.8.0",
    "isort==5.12.0",
    "types-PyYAML==6.0.12.10",
    "pre-commit==3.3.2",
    "pylint==3.2.7",
]

test_packages = ["pytest==7.3.2"]

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="tsfb",
    version="0.1",
    description="VNPT Time series forecasting benchmarking library",
    long_description=readme,
    long_description_content_type="text/markdown",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=required_packages,
    python_requires=">=3.11",
    extras_require={
        "dev": required_packages + style_packages + test_packages,
        "test": required_packages + test_packages,
    },
)
