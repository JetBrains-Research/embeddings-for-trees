[![embeddings-for-trees](https://circleci.com/gh/JetBrains-Research/embeddings-for-trees.svg?style=svg)](https://app.circleci.com/pipelines/github/JetBrains-Research/embeddings-for-trees)

# Embeddings for trees
PyTorch Tree2Seq framework for developing and evaluating different algorithms for tree embedding.

## Requirements
You can use conda manager for creating virtual environment:
```bash
conda env create -f environment.yml
```
Also, there is `requirements.txt` file that describe all used python packages.

## Data preprocessing

`data_preprocessing` package contain steps for converting source code into abstract syntax trees.
Main tool for building such trees is [astminer](https://github.com/JetBrains-Research/astminer).
You should build this tool from sources and define the path to `.jar` in `data_preprocessing/main.py`.

Also, there are already converted data: 3 datasets with Java function ASTs for predicting their names.
Below you can find statistic for each dataset and link to download it:

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
    <td>~350 000</td>
    <td>459 104</td>
  </tr>
</tbody>
</table>

## Tree2Seq framework

Implementation based on [PyTorch](https://pytorch.org/docs/stable/torch.html) framework.
For supporting learning on graphs [DGL](https://www.dgl.ai/) library is used.

Framework consist of 3 main parts:
- Node embedding (`INodeEmbedding`) that embed tokens in nodes to vectors.
- Tree encoder (`ITreeEncoder`) that encode tree into a vector representation.
- Tree decoder (`ITreeDecoder`) that decode vector representation into target sequence.

Converting source tree into target sequence occurs in `Tree2Seq` class and represent sequence applying of interface realization.
If you want to add a new algorithm, you should realize class with inheritance from the needed interface, and then use it in the configuration.

### Configuration

Components configuration and required hyper parameters can be passed to the model throw `.json` files.
The examples of such files can be found in `config` directory, there are already prepared configs for training and evaluating
Tree-LSTM model on Java dataset.

Use can use prepared scripts to interact with model:
- Start training ChildSum Tree-LSTM with logging to [wandb](https://www.wandb.com/) service
```bash
python train.py configs/tree_lstm_childsum_java_small.json wandb
```
- Evaluate model:
```bash
python evaluate.py configs/evaluate.json
```
- Predict target sequence for given file with concrete model (predict function name by its body),
this step includes data preprocessing and model applying:
```bash
python interactive.py test.java model.pt
```

## Evaluation

Currently, there are implemented Tree-LSTM model with various modifications. For all of them, LSTM with Attention to subtrees
is used to decoder to target sequence (method name). For node embedding used type (internal nodes) and token (leafs) matrix embeddings.

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

## Contribution

Supporting different algorithms to encode and decode trees are the main key areas of improvement this framework.
So, if you have any suggestions or questions feel free to open issues or even create pull requests. 
