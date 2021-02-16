[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Github action: build](https://github.com/JetBrains-Research/embeddings-for-trees/workflows/Test/badge.svg)](https://github.com/JetBrains-Research/embeddings-for-trees/actions?query=workflow%3ATest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)


# Embeddings for trees
Set of PyTorch modules for developing and evaluating different algorithms for embedding trees.

## Requirements
You can install the dependencies by using the requirement list
```bash
pip install -r requirements.txt
```
Although it's better to install [PyTorch](https://pytorch.org/get-started/locally/) and [DGL](https://www.dgl.ai/pages/start.html)
manually for correct CUDA support.

## Data preprocessing
List of some useful tools for preprocessing source code into a dataset with ASTs:

- [ASTMiner](https://github.com/JetBrains-Research/astminer) -
  tool for mining raw ASTs and path-based representation of code using ANTLR, GumTree and other grammars.
- [PSIMiner](https://github.com/JetBrains-Research/psiminer) -
  tool for processing PSI trees from IntelliJ IDEA and creating labeled dataset from them.

## Model zoo
List of supported algorithms of tree embeddings with links to wiki guides:

- [TreeLSTM](https://github.com/JetBrains-Research/embeddings-for-trees/wiki/TreeLSTM-guide)

## Contribution

Supporting different algorithms of encoding and decoding trees is the key area of improvement for this framework.
If you have any suggestions or questions, feel free to open issues and create pull requests.
