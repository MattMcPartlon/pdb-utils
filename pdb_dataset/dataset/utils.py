from math import sqrt
import random
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from einops import rearrange


def window_sum(x, w):
    assert x.shape[-1] >= w, f"shape:{x.shape}, w:{w}"
    return x.unfold(0, w, 1).sum(dim=-1)


def get_contiguous_crop(crop_len: int, n_res: int) -> Tuple[int, int]:
    """Get a contiguous interval to crop"""
    if crop_len < 0:
        return 0, n_res
    start, end = 0, n_res
    start = random.randint(0, (end - crop_len)) if end > crop_len else start
    return start, min(end, start + crop_len)


def get_dimer_crop_lens(
    partition: List[Tensor], crop_len: int, min_len=100
) -> Tuple[int, int]:
    assert len(partition) == 2, f"{len(partition)}"
    l1, l2 = len(partition[0]), len(partition[1])
    if l1 + l2 <= crop_len or crop_len < 0:
        c1, c2 = l1, l2
    elif random.randint(0, 1) == 1:
        end = min(l1, max(crop_len - l2, min_len))
        c1 = random.randint(max(crop_len - l2, min(l1, min_len)), end)
        c2 = min(max(crop_len - c1, min_len), l2)
    else:
        end = min(l2, max(crop_len - l1, min_len))
        c2 = random.randint(max(crop_len - l1, min(l2, min_len)), end)
        c1 = min(max(crop_len - c2, min_len), l1)
    return c1, c2


def get_dimer_crop(partition: List[Tensor], crop_len: int, min_len=100) -> List[Tensor]:
    c1, c2 = get_dimer_crop_lens(partition, crop_len, min_len=min_len)
    mn1, mx1 = get_contiguous_crop(c1, len(partition[0]))
    mn2, mx2 = get_contiguous_crop(c2, len(partition[2]))
    return [partition[0][mn1:mx1], partition[1][mn2:mx2]]


def get_dimer_spatial_crop(
    partition, coords: List[Tensor], crop_len, min_len: int = 100, sigma=4
) -> List[Tensor]:
    assert (
        len(coords) == 2 and coords[0].ndim == 2
    ), f"{len(coords)}, {[type(x) for x in coords]}"
    l1, l2 = get_dimer_crop_lens(partition, crop_len=crop_len, min_len=min_len)
    scores = 1 / (
        1
        + torch.square(
            torch.cdist(
                coords[0],
                coords[1],
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            / sigma
        )
    )
    s1 = scores.sum(dim=1)
    s1 = window_sum(torch.exp(sigma * s1 / torch.max(s1)), w=l1).numpy()
    c1 = np.random.choice(len(s1), p=s1 / np.sum(s1))
    s2 = torch.sum(scores[c1: c1 + l1, :], dim=0)
    s2 = window_sum(torch.exp(sigma * s2 / torch.max(s2)), w=l2).numpy()
    c2 = np.random.choice(len(s2), p=s2 / np.sum(s2))
    return [partition[0][c1: c1 + l1], partition[1][c2: c2 + l2]]


def impute_cb(bb_coords: Tensor, bb_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Impute CB atom position"""
    cb_mask, cb_coords = torch.all(bb_mask, dim=-1), torch.zeros(
        bb_coords.shape[0], 1, 3, device=bb_coords.device
    )
    # gly_mask = torch.tensor([1 if s != "G" else 0 for s in self.seq]).bool()
    # cb_mask = cb_mask & gly_mask
    cb_coords[cb_mask] = _impute_cb(bb_coords[cb_mask]).unsqueeze(-2)
    cb_coords[~cb_mask] = bb_coords[:, [1]][~cb_mask]
    return cb_coords


def _impute_cb(bb_coords: Tensor) -> Tensor:
    """Imputes coordinates of beta carbon from tensor of residue coordinates
    :param bb_coords: shape (n,4,3) where dim=1 is N,CA,C coordinates.
    :return: imputed CB coordinates for each residue
    """
    assert bb_coords.shape[1] == 3 and bb_coords.ndim == 3, f"{bb_coords.shape}"
    bb_coords = rearrange(bb_coords, "n a c -> a n c")
    N, CA, C = bb_coords  # noqa
    n, c = N - CA, C - CA
    n_cross_c = torch.cross(n, c)
    t1 = sqrt(1 / 3) * (n_cross_c / torch.norm(n_cross_c, dim=-1, keepdim=True))
    n_plus_c = n + c
    t2 = sqrt(2 / 3) * (n_plus_c / torch.norm(n_plus_c, dim=-1, keepdim=True))
    return CA + (t1 + t2)


"""
def get_ab_ag_spatial_crop(ag_ab: Protein, crop_len: int):
    crop_len = min(crop_len, len(ag_ab))
    ca_coords = ag_ab["CA"]
    ab_coords = ca_coords[ag_ab.chain_indices[0]]
    ag_coords = ca_coords[ag_ab.chain_indices[1]]
    ab_len, ag_len = map(len, ag_ab.chain_indices)
    ag_cropped_size = max(10, crop_len - ab_len)
    nearest_res, _ = torch.min(
        torch.cdist(
            ab_coords, ag_coords, compute_mode="donot_use_mm_for_euclid_dist"
        ),
        dim=0,
    )
    nearest_res[nearest_res < 8] = 1
    nearest_res[nearest_res >= 8] = 0
    scores = window_sum(nearest_res, w=ag_cropped_size).numpy()
    scores = np.maximum(1, scores) * (len(scores) ** (-1 / 2))
    scores = np.exp(scores)
    c2 = np.random.choice(len(scores), p=scores / np.sum(scores))
    return [
        ag_ab.chain_indices[0],
        ag_ab.chain_indices[1][c2 : c2 + ag_cropped_size],
    ]


def is_homodimer(chain_1: Protein, chain_2: Protein, tol=2) -> bool:
    if len(chain_1) != len(chain_2):
        return False
    return get_rmsd(chain_1, chain_2) < tol


def get_tm(a: Protein, b: Protein) -> float:
    mask = a.valid_residue_mask & b.valid_residue_mask
    return (
        -TMLoss()
        .forward(
            a["CA"][mask].clone().unsqueeze(0),
            b["CA"][mask].clone().unsqueeze(0),
            align=True,
            reduce=True,
        )
        .item()
    )


def get_rmsd(a: Protein, b: Protein) -> float:
    mask = a.valid_residue_mask & b.valid_residue_mask
    ca1, ca2 = map(
        lambda x: rearrange(x["CA"][mask], "n c -> () n () c"), (a, b)
    )
    loss = CoordDeviationLoss()(ca1, ca2, coord_mask=None, align=True)
    return loss.mean().item()


def restrict_protein_to_aligned_residues(a: Protein, b: Protein):
    valid_res_mask_a = torch.any(a.atom_masks, dim=-1)
    valid_res_mask_b = torch.any(b.atom_masks, dim=-1)
    valid_mask = valid_res_mask_a & valid_res_mask_b
    valid_indices = torch.arange(valid_mask.numel())[valid_mask]
    a, b = map(lambda x: _restrict_to_indices(x, valid_indices), (a, b))
    return a, b


def _restrict_to_indices(a: Protein, idxs: Tensor) -> Protein:
    sec_struct = None
    if exists(a.sec_struct):
        sec_struct = "".join([a.sec_struct[i] for i in idxs])
    return Protein(
        atom_coords=a.atom_coords[idxs],
        atom_masks=a.atom_masks[idxs],
        atom_tys=a.atom_tys,
        seq="".join([a.seq[int(i)] for i in idxs]),
        name=a._name,
        res_ids=[a.res_ids[0][idxs]],
        sec_struct=sec_struct,
    )
"""
