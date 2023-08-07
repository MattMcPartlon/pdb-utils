from typing import Optional, Callable
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pdb_dataset.helpers import exists
import lightning as L


def default(x, y):
    return x if exists(x) else y


class LigntningPDBDataset(L.LightningDataModule):
    """LightningDataModule wrapper for PDB dataset

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        train_data_conf: Optional[DictConfig] = None,
        val_data_conf: Optional[DictConfig] = None,
        test_data_conf: Optional[DictConfig] = None,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.train_data_conf = default(train_data_conf, OmegaConf.create({}))
        self.val_data_conf = default(val_data_conf, OmegaConf.create({}))
        self.test_data_conf = default(test_data_conf, OmegaConf.create({}))
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.collate_fn = collate_fn

    def _init_dataset(self, conf):
        if exists(conf.get("dataset")):
            return instantiate(conf.dataset)
        return None

    def _init_dataloader(self, conf, dataset):
        if exists(dataset):
            assert exists(
                conf.get("dataloader")
            ), "must specify dataloader in data config"
            return instantiate(
                conf.dataloader,
                dataset=dataset,
                collate_fn=default(self.collate_fn, dataset.collate_fn),
            )
        return None

    # no need for prepare_data
    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            if stage == "fit" or stage is None:
                self.data_train = self._init_dataset(self.train_data_conf)
                self.data_val = self._init_dataset(self.val_data_conf)

        if not self.data_test:
            if stage == "test" or stage is None:
                self.data_test = self._init_dataset(self.test_data_conf)

    def train_dataloader(self):
        return self._init_dataloader(self.train_data_conf, self.data_train)

    def val_dataloader(self):
        return self._init_dataloader(self.val_data_conf, self.data_val)

    def test_dataloader(self):
        return self._init_dataloader(self.test_data_conf, self.data_test)
