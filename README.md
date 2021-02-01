[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Github action: build](https://github.com/JetBrains-Research/embeddings-for-trees/workflows/Test/badge.svg)](https://github.com/JetBrains-Research/embeddings-for-trees/actions?query=workflow%3ATest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Embeddings for trees
PyTorch Tree2Seq framework for developing and evaluating different algorithms for embedding trees.

## Requirements
You can use conda manager to create a virtual environment:
```bash
conda env create -f environment.yml
```
Alternatively, you can install the dependencies manually,
but there may be problems during installation, because in Pypi and Conda some packages have different versions.
```bash
pip install -r requirements.txt
```

## Data preprocessing

The `data_preprocessing` package contains all the necessary steps to convert the source code into abstract syntax trees.
The main tool for building such trees is [astminer](https://github.com/JetBrains-Research/astminer).
You should build this tool from sources and define the path to `.jar` in `data_preprocessing/main.py`.

Also, you can use the data that is already converted: three datasets with Java function ASTs for predicting their names.
Below you can find their parameters and the download links.

<table>
<thead>
  <tr>
    <th rowspan="2">Dataset name</th>
    <th rowspan="2">Link</th>
    <th colspan="3">Number of projects</th>
    <th colspan="3">Number of functions</th>
  </tr>
  <tr>
    <td>train</td>
    <td>val</td>
    <td>test</td>
    <td>train</td>
    <td>val</td>
    <td>test</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Java-small</td>
    <td><a href="https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/java-ast-methods/java-ast-methods-small.tar.gz">s3 link</a></td>
    <td>10</td>
    <td>1</td>
    <td>1</td>
    <td>780 970</td>
    <td>32 654</td>
    <td>59 381</td>
  </tr>
  <tr>
    <td>Java-medium</td>
    <td><a href="https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/java-ast-methods/java-ast-methods-medium.tar.gz">s3 link</a></td>
    <td>800</td>
    <td>100</td>
    <td>96</td>
    <td>3 341 148</td>
    <td>457 527</td>
    <td>450 260</td>
  </tr>
  <tr>
    <td>Java-large</td>
    <td><a href="https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/java-ast-methods/java-ast-methods-large.tar.gz">s3 link</a></td>
    <td>8999</td>
    <td>250</td>
    <td>307</td>
    <td>16 794 775</td>
    <td>352 898</td>
    <td>459 104</td>
  </tr>
</tbody>
</table>

## Tree2Seq framework

The implementation is based on the [PyTorch](https://pytorch.org/docs/stable/torch.html) framework,
and we also use [DGL](https://www.dgl.ai/) to support learning on graphs.

The framework consists of three main parts:
- Node embedding (`INodeEmbedding`) embeds tokens on nodes to vectors.
- Tree encoder (`ITreeEncoder`) encodes trees into a vector representation.
- Tree decoder (`ITreeDecoder`) decodes the vector representation into a target sequence.

Converting the source tree into a target sequence is carried out in `Tree2Seq` class and consists in a consecutive applying of the interface realization.
If you want to add a new algorithm, you need to create a class with inheritance from the necessary interface, and then use it in a configuration.

### Configuration

The configuration of the components and required hyperparameters can be passed to the model via `.json` files.
The examples of such files can be found in `config` directory, prepared configs for training and evaluating the Tree-LSTM model on Java datasets are already present there.

You can use these scripts to interact with models:
- Start training [ChildSum Tree-LSTM](https://arxiv.org/abs/1503.00075) with logging to [wandb](https://www.wandb.com/) service:
```bash
python train.py configs/tree_lstm_childsum_java_small.json wandb
```
- Evaluate the model:
```bash
python evaluate.py configs/evaluate.json
```
- Predict a target sequence for a given file with concrete model (predict the function name by its body),
this step includes the preprocessing of data and the applying of the model:
```bash
python interactive.py test.java model.pt
```

## Evaluation

Currently, several Tree-LSTM models with various modifications are implemented.
For all of them, LSTM with Attention to subtrees is used to decode the representation into a target sequence (a method name).
As for the node embedding, the matrix embeddings for types (internal nodes) and tokens (leafs) are employed.

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Java-small</th>
    <th colspan="3">Java-medium</th>
  </tr>
  <tr>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-score</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-score</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ChildSum Tree-LSTM</td>
    <td>47.12</td>
    <td>30.27</td>
    <td>36.86</td>
    <td>56.11</td>
    <td>38.02</td>
    <td>45.33</td>
  </tr>
  <tr>
    <td>Attn. Tree-LSTM</td>
    <td>48.07</td>
    <td>30.44</td>
    <td>37.27</td>
    <td>56.52</td>
    <td>38.16</td>
    <td>45.53</td>
  </tr>
  <tr>
    <td>Sequence Tree-LSTM</td>
    <td>47.65</td>
    <td>30.53</td>
    <td>37.21</td>
    <td>56.84</td>
    <td>38.70</td>
    <td>46.05</td>
  </tr>
  <tr>
    <td>Conv. Tree-LSTM</td>
    <td>47.29</td>
    <td>30.06</td>
    <td>36.76</td>
    <td>56.97</td>
    <td>38.86</td>
    <td>46.20</td>
  </tr>
</tbody>
</table>

Information about modifications can be found in this [work](https://drive.google.com/file/d/1Mu-b0PqNgOCN_VC2s11LGEsBCaVg2rCt/view?usp=sharing) (russian language)

## Contribution

Supporting different algorithms of encoding and decoding trees is the key area of improvement for this framework.
If you have any suggestions or questions, feel free to open issues and create pull requests.
