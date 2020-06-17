from dataclasses import dataclass
from typing import List


@dataclass(init=False)
class IDatasetInfo:
    name: str = None
    url: str = None
    holdout_folders: List[str] = None
    language: str = None
    astminer_params: List[str] = None


@dataclass(init=False)
class _JavaDataset(IDatasetInfo):
    dataset_url = 'https://s3.amazonaws.com/code2seq/datasets/{}.tar.gz'
    holdout_folders = ['training', 'validation', 'test']
    language = "java"

    @property
    def url(self):
        return self.dataset_url.format(self.name)

    @property
    def astminer_params(self):
        return ['--storage', 'dot', '--granularity', 'method', '--lang', self.language, '--hide-method-name',
                '--split-tokens', '--java-parser', 'gumtree', '--filter-modifiers', 'abstract', '--remove-constructors',
                '--remove-nodes', 'Javadoc']


@dataclass(init=False)
class JavaSmallDataset(_JavaDataset):
    name = "java-small"


@dataclass(init=False)
class JavaMediumDataset(_JavaDataset):
    name = "java-med"


@dataclass(init=False)
class JavaLargeDataset(_JavaDataset):
    name = "java-large"


@dataclass(init=False)
class JavaTestDataset(_JavaDataset):
    name = "java-test"

    @property
    def url(self):
        raise NotImplementedError("There is no url for test dataset")
