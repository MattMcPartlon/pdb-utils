import urllib
import os
import sys
from typing import Optional, List, Tuple
from Bio.PDB.Polypeptide import one_to_three
from pdb_dataset.helpers import default
from pdb_dataset.residue_constants import BB_ATOMS, SC_ATOMS

atom_order = BB_ATOMS + sorted(SC_ATOMS)


def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)  # noqa
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None


def atom_to_id(atom_ty):
    """convert atom name to atom ID"""
    return atom_ty.upper()[0]


def format_coords(coords):
    """Format coordinates for pdb writer"""
    coords = list(map(lambda x: format(x, ".3f"), coords))
    space0 = make_space(12 - len(str(coords[0])))
    space1 = make_space(8 - len(str(coords[1])))
    space2 = make_space(8 - len(str(coords[2])))
    return f"{space0}{coords[0]}{space1}{coords[1]}{space2}{coords[2]}"


def make_space(spaces):
    return "".join([" "] * spaces)


def space_for_idx(idx):
    return make_space(7 - len(str(idx)))


def space_for_res_ty(atom_ty):
    return make_space(4 - len(atom_ty))


def space_for_res_idx(res_idx):
    return make_space(4 - len(str(res_idx)))


def add_ss(ss_string, seq, chain_ids):
    pdb_ss_string = ""
    SPACE = "            "
    blocks, labels = get_ss_blocks(ss_string)
    for block, label in zip(blocks, labels):
        if label == "E":
            s, e = block[0], block[-1]
            pdb_ss_string += f"SHEET{SPACE} {one_to_three(seq[s])} {chain_ids[s]} "
            pdb_ss_string += (
                f"{s}{space_for_res_idx(s)} {one_to_three(seq[e])}",
                f" {chain_ids[e]} {e}{space_for_idx(e)}\n",
            )


def get_ss_blocks(sec_struc: str) -> Tuple[List[List[int]], List[str]]:
    """Partition secondary structure into contiguous blocks

    Returns:
            ss_blocks
            where ss_blocks[i] = index of residues constituting block i
            ss_labels
            where ss_labels[i] = the secondary structure label of residues in block i
    """
    ss_blocks, ss_block, block_labels = [], [0], []
    for i in range(1, len(sec_struc)):
        if sec_struc[i] == sec_struc[i - 1]:
            ss_block.append(i)
        else:
            ss_blocks.append(ss_block)
            block_labels.append(sec_struc[i - 1])
            ss_block = [i]
    ss_blocks.append(ss_block)
    block_labels.append(sec_struc[-1])
    return ss_blocks, block_labels


def format_line(
    res_ty: str,
    atom_ty: str,
    coords: List,
    res_idx: int,
    atom_idx: int,
    chain_id: str,
) -> str:
    """Format single line for pdb atom entry"""
    res_ty = one_to_three(res_ty)
    line = f"ATOM{space_for_idx(atom_idx)}{atom_idx}  {atom_ty}"
    line += f"{space_for_res_ty(atom_ty)}{res_ty}"
    line += f" {chain_id}{space_for_res_idx(res_idx)}{res_idx}"
    line += format_coords(coords)
    line += f"  1.00  0.00           {atom_to_id(atom_ty)}\n"
    return line


def order_atoms(atom_tys):
    ordered = []
    for a in atom_order:
        if a in atom_tys:
            ordered.append(a)
    return ordered


def write_pdb(
    coord_dict,
    seq,
    out_path,
    chain_ids: Optional[List[str]] = None,
    res_idxs: Optional[List[int]] = None,
    header: str = "",
):
    assert out_path.endswith(".pdb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header += "" if header.endswith("\n") else "\n"
    pdb_string = header
    chain_ids = default(chain_ids, ["A"] * len(seq))
    last_chain = chain_ids[0]
    res_idx, atom_idx = 0, 0
    for res, atoms, chain in zip(seq, coord_dict, chain_ids):
        if chain != last_chain:
            last_chain = chain
            pdb_string += "TER\n"
            res_idx, atom_idx = 0, 0
        res_idx, atom_idx = res_idx + 1, atom_idx + 1
        for atom_ty in order_atoms(atoms):
            coords = atoms[atom_ty]
            pdb_string += format_line(
                res_ty=res,
                atom_ty=atom_ty,
                coords=coords,
                res_idx=int(res_idx),
                atom_idx=atom_idx,
                chain_id=chain,
            )

    with open(out_path, "w+") as f:
        f.write(pdb_string + "TER")
