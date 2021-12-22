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
This section contains information about implemented models.

### TreeLSTM

PyTorch reimplementation of ChildSum version of TreeLSTM presented by Tai et al. in
["Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"](https://arxiv.org/abs/1503.00075).

There are 3 different variantions of these model:
1. Classic — ChildSum TreeLSTM model for code AST with LSTM+attention decoder that was wisely used in other works
To train or test model use:
```shell
python -m embeddings_for_trees.treelstm train/test -c $CONFIG_PATH
```
2. Typed TreeLSTM — the same classic model, but works with typed AST that were introduced in [`PSIMiner`](https://arxiv.org/abs/2103.12778)
To train or test model use:
```shell
python -m embeddings_for_trees.typed_treelstm train/test -c $CONFIG_PATH
```
3. TreeLSTM with pointers — ChildSum TreeLSTM model with pointer network to AST leaves for decoding
To train or test model use:
```shell
python -m embeddings_for_trees.treelstm_ptr train/test -c $CONFIG_PATH
```

For config examples see [`config`](`./config`) directory.
Classic model and model with pointer network share configurations, while typed model requires specification for token types vocabulary.

## Contribution

Supporting different algorithms of encoding and decoding trees is the key area of improvement for this framework.
If you have any suggestions or questions, feel free to open issues and create pull requests.
