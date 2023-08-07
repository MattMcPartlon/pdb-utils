"""Protein Complex dataset"""
# pylint: disable=bare-except
from typing import Optional, List, Dict
import torch
from functools import lru_cache
from collections import defaultdict
from einops import repeat
from functools import partial

from pdb_dataset.dataset.dataset_abc import ProteinDatasetABC
from pdb_dataset.helpers import default
from pdb_dataset.io.pdb_utils import extract_atom_coords_n_mask_tensors
from pdb_dataset.dataset.utils import (
    impute_cb as impute_beta_carbon,
)
from pdb_dataset.dataset.collate_utils import default_collate
import pdb_dataset.residue_constants as rc


def cast_defaultdict(d):
    if isinstance(d, dict) or isinstance(d, defaultdict):
        return {k: cast_defaultdict(v) for k, v in d.items()}
    return d


class ProteinDataset(ProteinDatasetABC):
    """Generic protein dataset"""

    def __init__(
        self,
        pdb_list: str,
        decoy_folder: str,
        target_folder: str,
        seq_folder: Optional[str],
        raise_exceptions: bool,
        cluster_list: Optional[str] = None,
        shuffle: bool = True,
        atom_tys: Optional[List[str]] = None,
        crop_len: int = -1,
        load_sec_structure: bool = False,
        impute_cb: bool = True,
    ):
        super(ProteinDataset, self).__init__(
            model_list=pdb_list,
            decoy_folder=decoy_folder,
            target_folder=target_folder,
            seq_folder=seq_folder,
            raise_exceptions=raise_exceptions,
            shuffle=shuffle,
            cluster_list=cluster_list,
        )
        self.crop_len = crop_len
        self.atom_tys = default(atom_tys, rc.ALL_ATOMS)
        self.load_sec_structure = load_sec_structure
        self.default_atom_idx = min(len(self.atom_tys) - 1, 1)
        self.impute_cb = impute_cb

    @lru_cache(maxsize=1)
    def backbone_atom_tensor(self):
        valid_bb_atoms = set(self.atom_tys).intersection(set(rc.BB_ATOMS))
        return torch.tensor([self.atom_tys.index(x) for x in valid_bb_atoms]).long()

    @lru_cache(maxsize=1)
    def aa_to_canonical_atom_mask(self):
        msk = torch.zeros(21, len(self.atom_tys))
        for aa_idx in range(20):
            aa = rc.INDEX_TO_AA_ONE[aa_idx]
            for atom_idx, atom in enumerate(self.atom_tys):
                if atom in rc.BB_ATOMS:
                    msk[aa_idx, atom_idx] = 1
                elif atom in rc.AA_TO_SC_ATOMS[aa]:
                    msk[aa_idx, atom_idx] = 1
        return msk.bool()

    def append_chain_to_data(self, data_dict: Dict, pdb_path: str, seq: str) -> Dict:
        # target coords and mask
        crds, mask = extract_atom_coords_n_mask_tensors(
            seq, pdb_path=pdb_path, atom_tys=self.atom_tys
        )
        seq_encoding = torch.tensor([rc.AA_TO_INDEX[x] for x in seq])
        canonical_mask = self.aa_to_canonical_atom_mask()[seq_encoding]
        atom_mask = mask & canonical_mask
        ca_crds = repeat(
            crds[:, self.default_atom_idx, :],
            "i c -> i a c",
            a=len(self.atom_tys),
        )
        crds[~atom_mask] = ca_crds[~atom_mask]

        data_dict["sequence"].append(seq)

        data_dict["tokenized_sequence"].append(seq_encoding.long())
        data_dict["atom_mask"].append(atom_mask)

        data_dict["coordinates"].append(crds)
        data_dict["residue_mask"].append(
            torch.all(mask[:, self.backbone_atom_tensor()], dim=-1)
        )
        return data_dict

    def get_item_from_pdbs_n_seq(
        self,
        seq_paths: List[Optional[str]],
        decoy_pdb_paths: List[Optional[str]],
        target_pdb_paths: List[Optional[str]],
    ) -> Dict:
        """Load data given native and decoy pdb paths and sequence path

        Output is a dictionary with keys "target" and "decoy"
        Typically, the data within the "decoy" sub-dict is used
        to generate features/train the model, and the "target"
        sub dict is used as ground-truth labels.

        For example, this format allows the user to do things like
        train on unbound chain conformations (decoy data) and predict bound
        conformations (native data).
        """
        batch_data = dict(
            metadata=dict(
                atom_tys=self.atom_tys,
                decoy_pdb_paths=decoy_pdb_paths,
                target_pdb_paths=target_pdb_paths,
                seq_paths=seq_paths,
            )
        )
        target_data, decoy_data = defaultdict(list), defaultdict(list)
        for seq_path, decoy_pdb, target_pdb in zip(
            seq_paths, decoy_pdb_paths, target_pdb_paths
        ):
            seq = self.safe_load_sequence(seq_path, decoy_pdb)
            # TODO: Do not currently support case where target
            # and decoy have different underlying sequences.
            decoy_data = self.append_chain_to_data(decoy_data, decoy_pdb, seq)
            if "CB" in self.atom_tys and self.impute_cb:
                # CB is used to generate input features and may not
                #  be available at inference time
                for i in range(len(decoy_data["coordinates"])):
                    imputed_cb = impute_beta_carbon(
                        decoy_data["coordinates"][i][:, :3],
                        decoy_data["atom_mask"][i][:, :3],
                    )
                    decoy_data["coordinates"][i][
                        :, [self.atom_tys.index("CB")]
                    ] = imputed_cb

            target_data = self.append_chain_to_data(target_data, target_pdb, seq)

        batch_data["decoy"], batch_data["target"] = map(
            cast_defaultdict, (decoy_data, target_data)
        )
        batch_data["decoy"]["pdb_paths"] = decoy_pdb_paths
        batch_data["target"]["pdb_paths"] = target_pdb_paths
        return batch_data

    @property
    def collate_fn(self):
        return partial(
            default_collate,
            max_len=self.crop_len,
            atom_idx=self.default_atom_idx,
        )
