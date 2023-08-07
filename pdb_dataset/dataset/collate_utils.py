"""Protein Complex dataset"""
# pylint: disable=bare-except
from typing import Optional, List, Dict
import os
import torch

from pdb_dataset.helpers import exists
from pdb_dataset.dataset.utils import (
    get_contiguous_crop,
    get_dimer_spatial_crop,
)
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from typeguard import typechecked


# Example model.collate function:
@typechecked
def default_collate(
    batch: List[Optional[Dict]],
    pre_collate: bool = True,
    crop_len: int = -1,
    default_atom_idx: int = 1,
) -> Dict:
    """
    "decoy" and "target" contain the following <key,value> mappings:

    sequence: (b,n, torch.long) amino acid identities according to
        common.protein_constants
    coords: (b,n,a,3) atom coordinates
    residue_mask: (b,n,torch.bool) mask indicating which residues are valid
    atom_mask: (b,n,torch.bool) mask indicating which atoms are valid
    chain_ids: (b, n, torch.long) chain_ids[b,i] is the index of the chain
        which residue i in the b-th example belongs to (indexed from 0)
    pdb_ids: (List[List[str]]) pdb_ids[i] gives the pdb name for each chain
        in the ith batch element.

    We also use this function to generate feature encodings for
    the batch in parallel, on cpu
    """
    batch = list(filter(exists, batch))
    if len(batch) == 0:
        return {}  # possible if error loading pdb

    if pre_collate:
        batch = basic_collate(
            batch,
            max_len=crop_len,
            atom_idx=default_atom_idx,
        )

    batch_data = dict(metadata=batch[0]["metadata"])
    batch_data["metadata"]["atom_posns"] = {
        a: i for i, a in enumerate(batch[0]["metadata"]["atom_tys"])
    }

    batch_data["decoy"] = _concat_data(batch, "decoy")
    batch_data["target"] = _concat_data(batch, "target")

    # also store the original batch data, before crop and collate
    batch_data["raw"] = batch

    return batch_data


@typechecked
def basic_collate(
    batch: List[Optional[Dict]], max_len: int = -1, atom_idx: int = 0
) -> List[Dict]:
    batch = filter(exists, batch)

    batch = list(
        map(
            lambda x: crop_example(
                x,
                max_len=max_len,
                default_atom_idx=atom_idx,
            ),
            batch,
        )
    )
    batch = list(map(augment_with_feature_args, batch))
    return batch


@typechecked
def crop_example(
    single_example: Dict,
    max_len: int = -1,
    default_atom_idx: int = 0,
) -> Dict:
    chain_lens = list(map(lambda x: x.shape[0], single_example["decoy"]["coordinates"]))
    bounds = [(0, cl) for cl in chain_lens]
    if sum(chain_lens) > max_len > 0:
        # perform crop
        chain_lens = list(
            map(lambda x: x.shape[0], single_example["decoy"]["coordinates"])
        )
        is_monomer = len(chain_lens) == 1
        keys_to_crop = [
            "sequence",
            "tokenized_sequence",
            "atom_mask",
            "residue_mask",
            "coordinates",
        ]
        if is_monomer:
            start, end = get_contiguous_crop(max_len, n_res=chain_lens[0])
            bounds = [(start, end)]
        else:
            assert (
                len(chain_lens) == 2
            ), "more than two chains not currently supported for cropping :/"
            crop_idxs = get_dimer_spatial_crop(
                partition=[
                    torch.arange(chain_lens[0]),
                    torch.arange(chain_lens[1]),
                ],
                coords=list(
                    map(
                        lambda x: x[..., default_atom_idx, :],
                        single_example["target"]["coordinates"],
                    ),
                ),
                crop_len=max_len,
            )
            bounds = list(map(lambda x: (x[0], x[-1]), crop_idxs))

        for idx, (start, end) in enumerate(bounds):
            for ex in ["decoy", "target"]:
                for k in keys_to_crop:
                    single_example[ex][k][idx] = single_example[ex][k][idx][start:end]

    single_example["decoy"]["crop_positions"] = bounds
    single_example["target"]["crop_positions"] = bounds
    return single_example


def augment_with_feature_args(single_example: Dict) -> Dict:
    for ty in ["decoy", "target"]:
        curr_data = single_example[ty]
        cat_key = lambda key, dat=single_example[ty]: torch.cat(dat[key], dim=0)  # noqa
        args = dict(
            coords=cat_key("coordinates"),
            residue_mask=cat_key("residue_mask").bool(),
            atom_mask=cat_key("atom_mask").bool(),
            sequence=cat_key("tokenized_sequence").long(),
            sequence_string="".join(curr_data["sequence"]),
            residue_indices=torch.cat(
                [torch.arange(v.shape[0]) for v in curr_data["atom_mask"]]
            ),
            chain_ids=torch.cat(
                [
                    idx * torch.ones(v.shape[0])
                    for idx, v in enumerate(curr_data["atom_mask"])
                ]
            ).long(),
            pdb_ids=[
                list(
                    map(
                        os.path.basename,
                        single_example[ty]["pdb_paths"],
                    )
                )
            ],
            target_coords=cat_key("coordinates", dat=single_example["target"]),
        )
        single_example[ty]["feature_args"] = args
    return single_example


def _concat_data(batch_data: List[Dict], key: str) -> Dict:
    feature_args = defaultdict(list)
    for entry in batch_data:
        tmp = entry[key]["feature_args"]
        for k in tmp:
            feature_args[k].append(tmp[k])
    cat_batch = (
        lambda x: pad_sequence(x, batch_first=True) if torch.is_tensor(x[0]) else x
    )
    concatenated = {k: cat_batch(v) for k, v in feature_args.items()}
    return concatenated


def collate_esm_input(batch: Dict):
    batch
