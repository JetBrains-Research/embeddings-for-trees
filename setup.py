from setuptools import setup, find_packages

VERSION = "1.1.0"

with open("README.md") as readme_file:
    readme = readme_file.read()

install_requires = [
    "torch>=1.10.0",
    "pytorch-lightning~=1.5.0",
    "wandb~=0.12.0",
    "omegaconf~=2.1.1",
    "commode-utils>=0.4.1",
    "dgl~=0.6.1",
]

setup_args = dict(
    name="embeddings-for-trees",
    version=VERSION,
    description="Set of PyTorch modules for developing and evaluating different algorithms for embedding trees.",
    long_description_content_type="text/markdown",
    long_description=readme,
    install_requires=install_requires,
    license="MIT",
    packages=find_packages(),
    author="Egor Spirin",
    author_email="spirin.egor@gmail.com",
    keywords=["treelstm", "tree-lstm", "embeddings-for-trees", "pytorch", "pytorch-lightning", "ml4code", "ml4se"],
    url="https://github.com/JetBrains-Research/embeddings-for-trees",
    download_url="https://pypi.org/project/embeddings-for-trees/",
)

if __name__ == "__main__":
    setup(**setup_args)
