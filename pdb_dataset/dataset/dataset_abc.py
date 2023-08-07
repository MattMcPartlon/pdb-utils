"""Baseline protein dataset"""
# pylint: disable=bare-except
import torch
from abc import abstractmethod
from typing import Optional, Tuple, List, Union, Dict
import os
import numpy as np
from torch.utils.data import Dataset
import random
from functools import lru_cache
import logging

from pdb_dataset.io.seq_utils import load_fasta_file
from pdb_dataset.helpers import exists, default
from pdb_dataset.io.pdb_utils import (
    extract_pdb_seq_from_pdb_file,
)

log = logging.getLogger()

torch.multiprocessing.set_sharing_strategy("file_system")
g = torch.Generator()
g.manual_seed(0)  # noqa


def seed_worker(worker_id=None):  # noqa
    worker_seed = default(worker_id, torch.initial_seed() % 2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_model_list(list_path: str, max_to_load: int = -1) -> List[List[str]]:
    """Loads a model list"""
    all_data = []
    with open(list_path, "r") as f:
        for i, line in enumerate(f):
            targets = line.strip().split()
            if len(targets) > 0 and len(targets[0]) > 1:
                dat = [t[:-4] if t.endswith(".pdb") else t for t in targets]
                all_data.append([dat])
            if i > max_to_load > 0:
                break
    return all_data


def load_clusters(cluster_list_path: str, max_to_load: int = -1) -> List[List[str]]:
    """Loads data from a cluster list list"""
    all_data = []
    with open(cluster_list_path, "r") as f:
        for i, cluster_path in enumerate(f):
            cluster_data = []
            with open(cluster_path.strip()) as cluster:
                for line in cluster:
                    targets = line.strip().split()
                if len(targets) > 0 and len(targets[0]) > 1:
                    dat = [t[:-4] if t.endswith(".pdb") else t for t in targets]
                    cluster_data.append(dat)
                if i > max_to_load > 0:
                    break
            if len(cluster_data) > 0:
                all_data.append(cluster_data)
    return all_data


def item(x: Union[str, List[str]]) -> Union[str, List[str]]:
    """Return first item in list if len==1 else whole list."""
    if isinstance(x, str):
        return x
    return x if len(x) > 1 else x[0]


class ModelList:
    """

    If cluster list is not None,
    """

    def __init__(
        self,
        model_list_path: str,
        target_folder: str,
        decoy_folder: str,
        seq_folder: str,
        cluster_list: Optional[str] = None,
    ):
        if exists(cluster_list):
            self.model_list = load_clusters(cluster_list)
        else:
            self.model_list = load_model_list(model_list_path)
        self.target_folder, self.decoy_fldr, self.seq_fldr = (
            target_folder,
            decoy_folder,
            seq_folder,
        )
        self.idxs = np.arange(len(self.model_list))
        self.cluster_list = cluster_list

    def shuffle(self):
        """Shuffles the model list"""
        np.random.shuffle(self.idxs)

    @lru_cache(maxsize=1)
    def max_cluster_size(self):
        return max(map(len, self.model_list))

    @staticmethod
    def _get_path(
        entry: List[str], fldr: str, exts: Optional[List]
    ) -> List[Optional[str]]:
        """get pdb path for list entries with source folder fldr"""
        exts = default(exts, [])
        target_paths = []
        for target in entry:
            added = False
            if target.lower() == "none":
                added = True
                target_paths.append(None)
            for tgt in [target] + [f"{target}.{ext}" for ext in exts]:
                path = os.path.join(fldr, tgt)
                if os.path.exists(path):
                    added = True
                    target_paths.append(path)
            if not added:
                raise Exception(f"no path found for\n" f"entry : {entry}\ndir: {fldr}")
        return target_paths

    def _get_native_pdb_path(self, entry) -> Union[str, List[str]]:
        """get native pdb path for list entry"""
        return self._get_path(entry, self.target_folder, ["pdb"])

    def _get_decoy_pdb_path(self, entry) -> str:
        """get decoy pdb path for list entry"""
        return self._get_path(entry, self.decoy_fldr, ["pdb"])

    def _get_seq_path(self, entry) -> Optional[str]:
        """get sequence path for list entry"""
        if not exists(self.seq_fldr):
            return [None] * len(entry)
        return self._get_path(entry, self.seq_fldr, ["fasta"])

    def get_entry(self, idx):
        """Gets given list entry"""
        cluster = self.model_list[self.idxs[idx]]
        return cluster[np.random.randint(0, len(cluster))]

    def __getitem__(
        self, idx: int
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[Exception]]:
        entry = self.get_entry(self.idxs[idx])
        native_pdbs, decoy_pdbs = [None] * len(entry), [None] * len(entry)
        seq_paths, exceptions = [None] * len(entry), []
        try:
            native_pdbs = self._get_native_pdb_path(entry)
        except Exception as e:
            exceptions.append(e)
        try:
            decoy_pdbs = self._get_decoy_pdb_path(entry)
        except Exception as e:
            exceptions.append(e)
        try:
            seq_paths = self._get_seq_path(entry)
        except Exception as e:  # noqa
            exceptions.append(e)

        return seq_paths, decoy_pdbs, native_pdbs, exceptions

    def __len__(self):
        return len(self.model_list)


class ProteinDatasetABC(Dataset):
    """Dataset base class"""

    def __init__(
        self,
        model_list: Optional[str],
        decoy_folder: str,
        target_folder: str,
        seq_folder: str,
        raise_exceptions: bool,
        crop_len: int = -1,
        shuffle: bool = True,
        cluster_list: Optional[str] = None,
    ):
        super().__init__()
        assert exists(model_list) or exists(cluster_list)
        self.model_list = ModelList(
            model_list,
            target_folder=target_folder,
            decoy_folder=decoy_folder,
            seq_folder=seq_folder,
            cluster_list=cluster_list,
        )
        if shuffle:
            self.shuffle()
        self.raise_exceptions = raise_exceptions
        self.crop_len = crop_len
        self.target_folder, self.decoy_folder = target_folder, decoy_folder

    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        (
            seq_paths,
            decoy_pdb_paths,
            target_pdb_paths,
            exceptions,
        ) = self.model_list[idx]

        # optionally raise exceptions
        if len(exceptions) > 0:
            msgs = [str(e) for e in exceptions]
            log.warn(f"caught exceptions {msgs} loading data")
            if self.raise_exceptions:
                raise exceptions[0]
        try:
            return self.get_item_from_pdbs_n_seq(
                seq_paths=seq_paths,
                decoy_pdb_paths=decoy_pdb_paths,
                target_pdb_paths=target_pdb_paths,
            )
        except Exception as e:  # noqa
            log.warn(f"Got exception : {e} in dataloader abc")
            if self.raise_exceptions:
                raise e
            # load another (valid) example at random
            return self[np.random.randint(0, len(self) - 1)]

    def shuffle(self) -> None:
        """Shuffle the dataset"""
        self.model_list.shuffle()

    @abstractmethod
    def get_item_from_pdbs_n_seq(
        self,
        seq_paths: List[Optional[str]],
        decoy_pdb_paths: List[Optional[str]],
        target_pdb_paths: List[Optional[str]],
    ) -> Dict:
        """Load data given native and decoy pdb paths and sequence path"""
        return

    @staticmethod
    @lru_cache(16)
    def safe_load_sequence(seq_path: Optional[str], pdb_path: str) -> str:
        """Loads sequence, either from fasta or given pdb file"""
        if exists(seq_path):
            pdbseqs = [load_fasta_file(seq_path)]
        else:
            pdbseqs, *_ = extract_pdb_seq_from_pdb_file(pdb_path)
        if len(pdbseqs) > 1:
            log.warn(f"Multiple chains found for pdb: {pdb_path}")
            pass
        return pdbseqs[0]

    def collate(self, batch: List[Optional[Dict]]) -> List[Dict]:
        return list(filter(exists, batch))
