import os

from setuptools import find_packages, setup

REQUIRED_PKGS = ["torch", "transformers"]

QUALITY_REQUIRE = ["black~=22.0", "flake8>=3.8.3", "isort>=5.0.0", "pyyaml>=5.3.1"]

setup(
    name="trfs_prealloc",
    version="0.0.1.dev0",
    description="HuggingFace community-driven open-source library of evaluation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HuggingFace Inc.",
    author_email="felix@huggingface.co",
    url="https://github.com/fxmarty/transformers-preallocate-kv-cache",
    download_url="https://github.com/fxmarty/transformers-preallocate-kv-cache",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=REQUIRED_PKGS,
    python_requires=">=3.8.0",
)
