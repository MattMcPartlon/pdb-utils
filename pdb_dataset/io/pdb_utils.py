"""Utility functions for working with pdbs

Adapted from Raptorx3DModelling/Common/PDBUtils.py
"""
import os
from typing import Dict, List, Tuple, Union, Optional
from functools import lru_cache
import numpy as np
import torch
from Bio import pairwise2
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa, three_to_one as _three_to_one
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.Align import substitution_matrices
from torch import Tensor
from collections import defaultdict


BLOSUM80 = substitution_matrices.load("BLOSUM80")


def three_to_one(x):
    return _three_to_one(x) if is_aa(x) else "X"


def default(x, y):
    return x if x is not None else y


class SubMat:
    """Wrapper Around BLOSUM80 subst. matrix for handling
    non-standard residue types
    """

    @staticmethod
    def subst_matrix_get(aa1, aa2):
        """Heavy penalty for non-standard aas"""
        if (aa1, aa2) in BLOSUM80:
            val = BLOSUM80[(aa1, aa2)]
        elif (aa2, aa1) in BLOSUM80:
            val = BLOSUM80[(aa2, aa1)]
        else:
            val = -1000
        return val

    def __contains__(self, *item):
        return True

    def __getitem__(self, item):
        return self.subst_matrix_get(*item)


class PDBExtractException(Exception):
    pass


def get_structure_parser(pdb_file: str) -> Union[PDBParser, MMCIFParser]:
    """gets a parser for the underlying pdb structure

    :param pdb_file: the file to obtain a structure parser for
    :return: structure parser for pdb input
    """
    is_pdb, is_cif = [pdb_file.endswith(x) for x in (".pdb", ".cif")]
    assert (
        is_pdb or is_cif
    ), f"ERROR: pdb file must have .cif or .pdb type, got {pdb_file}"
    return MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)


@lru_cache(maxsize=16)
def get_structure(pdbfile: str, name: str = None):
    """Get BIO.Structure object"""
    parser = get_structure_parser(pdbfile)
    name = default(name, os.path.basename(pdbfile))
    return parser.get_structure(name, pdbfile)


def extract_pdb_seq_from_residues(
    residues: List[Residue],
) -> Tuple[str, List[Residue]]:
    """
    extract a list of residues with valid 3D coordinates excluding
    non-standard amino acids
    returns the amino acid sequence as well as a list of residues
    with standard amino acids
    """
    residueList = list(filter(lambda r: is_aa(r, standard=True), residues))
    res_names = list(map(lambda x: x.get_resname(), residueList))
    pdbseq = "".join(list(map(three_to_one, res_names)))
    return pdbseq, residueList


def extract_pdb_seq_by_chain(structure: Structure) -> Tuple[List, ...]:
    """extract sequences and residue lists for each chain
    :return: pdbseqs, residue lists and also the chain objects
    """
    model = structure[0]
    pdbseqs, residueLists, chains = [], [], []
    for chain in model:
        residues = list(chain.get_residues())
        pdbseq, residueList = extract_pdb_seq_from_residues(residues)
        pdbseqs.append(pdbseq)
        residueLists.append(residueList)
        chains.append(chain)
    return pdbseqs, residueLists, chains


@lru_cache(maxsize=8)
def extract_pdb_seq_from_pdb_file(
    pdbfile: str, name: Optional[str] = None
) -> Tuple[List, ...]:
    """Extract sequences and residue lists from pdbfile for all the chains
    :param pdbfile: pdb file to extract from
    :param name: name for bio.pdb structure
    :return: lists of : pdbseqs, residueLists, chains from each
    chain in input pdb file
    """
    name = default(name, os.path.basename(pdbfile)[:-4])
    structure = get_structure(pdbfile=pdbfile, name=name)
    return extract_pdb_seq_by_chain(structure)


def calc_num_mismatches(alignment) -> Tuple[int, int]:
    """Calculate number of mismatches in sequence alignment

    :param alignment: sequence alignment(s)
    :return: number of mismatches in sequence alignment
    """
    S1, S2 = alignment[:2]
    numMisMatches = np.sum(
        [
            a != b
            for a, b in zip(S1, S2)
            if a != "-" and b != "-" and a != "X" and b != "X"
        ]
    )
    numMatches = np.sum([a == b for a, b in zip(S1, S2) if a != "-" and a != "X"])
    return int(numMisMatches), int(numMatches)


def alignment_to_mapping(alignment) -> List:
    """Convert sequence alignment to residue-wise mapping

    :param alignment: sequence alignment
    :return: mapping
    """
    S1, S2 = alignment[:2]
    # convert an aligned seq to a binary vector with 1 indicates
    # aligned and 0 gap.
    y = np.array([1 if a != "-" else 0 for a in S2])
    # get the position of each residue in the original sequence,
    # starting from 0.
    ycs = np.cumsum(y) - 1
    np.putmask(ycs, y == 0, -1)
    # map from the 1st seq to the 2nd one. set -1 for an unaligned residue
    # in the 1st sequence.
    mapping = [y0 for a, y0 in zip(S1, ycs) if a != "-"]
    return mapping


def map_seq_to_residue_list(
    sequence: str, pdbseq: str, residueList: List[Residue]
) -> Tuple[Optional[List], Optional[int], Optional[int]]:
    """map one query sequence to a list of PDB residues by sequence alignment
    pdbseq and residueList are generated by ExtractPDBSeq or
    ExtractPDBSeqByChain from a PDB file
    :param sequence:
    :param pdbseq:
    :param residueList:
    :return: seq2pdb mapping, numMisMatches and numMatches
    """
    # here we align PDB residues to query sequence instead
    # of query to PDB residues
    if pdbseq != sequence:
        alignments = pairwise2.align.localds(pdbseq, sequence, BLOSUM80, -5, -0.2)
    else:
        alignments = [pairwise2.Alignment(sequence, sequence, 100, 0, len(sequence))]
    if not bool(alignments):
        return None, None, None

    # find the alignment with the minimum difference
    diffs = []
    for alignment in alignments:
        mapping_pdb2seq, diff = alignment_to_mapping(alignment), 0
        for current_map, prev_map, current_residue, prev_residue in zip(
            mapping_pdb2seq[1:],
            mapping_pdb2seq[:-1],
            residueList[1:],
            residueList[:-1],
        ):
            # in principle, every PDB residue with valid 3D coordinates
            # shall appear in the query sequence. otherwise, apply a big penalty
            if current_map < 0:
                diff += 10
                continue

            if prev_map < 0:
                continue

            # calculate the difference of sequence separation in both
            # the PDB seq and the query seq. the smaller, the better
            current_id = current_residue.get_id()[1]
            prev_id = prev_residue.get_id()[1]
            id_diff = max(1, current_id - prev_id)
            map_diff = current_map - prev_map
            diff += abs(id_diff - map_diff)

        numMisMatches, numMatches = calc_num_mismatches(alignment)
        diffs.append(diff - numMatches)

    diffs = np.array(diffs)
    alignment = alignments[diffs.argmin()]

    numMisMatches, numMatches = calc_num_mismatches(alignment)

    # map from the query seq to pdb
    mapping_seq2pdb = alignment_to_mapping((alignment[1], alignment[0]))

    return mapping_seq2pdb, numMisMatches, numMatches


def map_seq_to_pdb(
    sequence,
    pdbfile,
    maxMisMatches=None,
    minMatchRatio=0.5,
):
    """Maps sequence to a pdb file,
      selecting the sequence from chain with best match.
    :param sequence: sequence (string)
    :param pdbfile: pdb file to map to
    :param maxMisMatches: max allowed number of mismatches
    :param minMatchRatio: the minimum ratio of matches on the query sequence
    :return: seq2pdb mapping, the pdb residue list, the pdb seq, the pdb chain,
     the number of mismtaches and matches
    """
    maxMisMatches = max(5, default(maxMisMatches, int(0.1 * len(sequence))))
    if not os.path.isfile(pdbfile):
        # pylint: disable-next=broad-exception-raised
        raise Exception("ERROR: the pdb file does not exist: ", pdbfile)

    # extract PDB sequences by chains
    pdbseqs, residueLists, chains = extract_pdb_seq_from_pdb_file(pdbfile)

    bestPDBSeq = None
    bestMapping = None
    bestResidueList = None
    bestChain = None
    minMisMatches = np.iinfo(np.int32).max
    maxMatches = np.iinfo(np.int32).min

    for pdbseq, residueList, chain in zip(pdbseqs, residueLists, chains):
        seq2pdb_mapping, numMisMatches, numMatches = map_seq_to_residue_list(
            sequence, pdbseq, residueList
        )
        if seq2pdb_mapping is None:
            continue
        if maxMatches < numMatches:
            # if numMisMatches < minMisMatches:
            bestMapping = seq2pdb_mapping
            minMisMatches = numMisMatches
            maxMatches = numMatches
            bestResidueList = residueList
            bestPDBSeq = pdbseq
            bestChain = chain

    if minMisMatches > maxMisMatches:
        print(
            f"ERROR: there are  {minMisMatches} mismatches between"
            f" the query sequence and PDB file: {pdbfile}\n"
            f"num residue : {len(sequence)}"
        )
        return None, None, None, None, None, None

    if maxMatches < min(30.0, minMatchRatio * len(sequence)):
        print(
            "ERROR: there are only  {maxMatches} matches on query sequence, "
            f"less than  {minMatchRatio} of its length from PDB file: {pdbfile}"
        )
        return None, None, None, None, None, None

    return (
        bestMapping,
        bestResidueList,
        bestPDBSeq,
        bestChain,
        minMisMatches,
        maxMatches,
    )


def extract_coords_by_mapping(seq2pdb_mapping, residueList, atom_tys):
    """Extract coordinates from residue list by sequence mapping
    :param sequence:
    :param seq2pdb_mapping:
    :param residueList:
    :param atoms:
    :return:
    """
    needed_atoms = [a.upper() for a in atom_tys]
    needed_atom_set = set(needed_atoms)
    atomCoordinates = []
    for aligned_pos in seq2pdb_mapping:
        coordinates = defaultdict(lambda: None)

        if aligned_pos >= 0:
            res = residueList[aligned_pos]
            for atom in res:
                atom_name = atom.get_id().upper()
                if atom_name in needed_atom_set:
                    coordinates[atom_name] = atom.get_vector()

        atomCoordinates.append(coordinates)

    return atomCoordinates


def extract_seq_from_pdb_n_chain_id(
    pdbfile: str, chain_id: str, name: str = None
) -> str:
    """Extract the sequence of a specific pdb chain"""

    name = default(name, os.path.basename(pdbfile)[:-4])
    structure = get_structure(pdbfile=pdbfile, name=name)
    model = structure[0]
    chain_ids = []
    for chain in model:
        chain_ids.append(chain.get_id())
        residues = chain.get_residues()
        if chain.get_id() == chain_id:
            pdbseq, _ = extract_pdb_seq_from_residues(residues)
            return pdbseq
    raise PDBExtractException(
        f"No chain with id {chain_id}, found chains: {[chain_ids]}"
    )


def extract_coords_from_seq_n_pdb(
    sequence,
    pdbfile,
    atom_tys,
    maxMisMatches=5,
    minMatchRatio=0.5,
):
    """
    :param sequence: sequence to map from
    :param pdbfile: pdb file to extract from
    :param atom_tys: atom types to extract coords for.
    :param maxMisMatches: maximum number of allowed mismatches
      in seq to pdb_seq alignment
    :param minMatchRatio: minimum allowed match ratio
    :return: tuple containining:
        (1) atom_coordinates : atom coordinates for each residue
          in the input sequence
        (2) pdb_seq : pdb sequence for chain which was mapped to
        (3) num_mismatches: number of mismatches in alignment
        (4) num_matches: number of matches in alignment
    """
    out = map_seq_to_pdb(
        sequence=sequence,
        pdbfile=pdbfile,
        maxMisMatches=maxMisMatches,
        minMatchRatio=minMatchRatio,
    )

    (
        seq2pdb_mapping,
        residueList,
        pdbseq,
        _,  # chain
        num_mismatches,
        num_matches,
    ) = out
    if seq2pdb_mapping is None:
        return None, None, None, None, None
    residueList = list(residueList)
    atom_coordinates = extract_coords_by_mapping(
        seq2pdb_mapping=seq2pdb_mapping,
        residueList=residueList,
        atom_tys=atom_tys,
    )
    res_ids = [r.get_id()[1] for r in residueList]
    return atom_coordinates, pdbseq, num_mismatches, num_matches, res_ids


def extract_atom_coords_n_mask_tensors(
    seq: Optional[str],
    pdb_path: str,
    atom_tys: List[str],
) -> Union[Tuple[Tensor, Tensor, Tensor, str], Tuple[Tensor, Tensor, str]]:
    """Extracts
    :param seq: sequence to map from
    :param pdb_path: pdb path to extract coordinates from
    :param atom_tys: atom types to extract coordinates for
    :return: Tuple containing
        (1) coords: Tensor of shape (n,a,3) containing atom coordinates for
        each atom type (1..a) and each residue 1..n in the input sequence
        (2) mask: Tensor of shape (n,a) indicating whether valid coordinates
        were obtained for atom types for each atom in atom_tys (1..a) and each
        residue (1..n) in the input sequence
    """
    assert seq is not None, "must provide sequence to extract coords and masks"

    out = extract_coords_from_seq_n_pdb(
        sequence=seq,
        pdbfile=pdb_path,
        atom_tys=atom_tys,
    )

    atom_coords, _, numMisMatches, *_ = out
    atom_mask = None
    if atom_coords is not None:
        if numMisMatches > 5:
            print(
                f"WARNING: got {numMisMatches} ",
                f"mismatches mapping seq. to pdb\n{pdb_path}",
            )
        atom_coords, atom_mask = _get_coord_n_mask_tensors(atom_coords, atom_tys)
    return atom_coords, atom_mask


def _get_coord_n_mask_tensors(
    atom_coords: List[Dict[str, np.ndarray]], atom_tys: List[str]
) -> Tuple[Tensor, Tensor]:
    """Retrieves coord and mask tensors from output of
    extract_coords_from_seq_n_pdb(...).

    :param atom_coords: List of dictionaries. each dict mapping from atom type
      to atom coordinates.
    :param atom_tys: the atom types to extract coordinates for.
    :return: Tuple containing
        (1) coords: Tensor of shape (n,a,3) containing atom coordinates for
        each atom type (1...a) and each residue 1..n in the input sequence.
        (2) mask: Tensor of shape (n,a) indicating whether valid coordinates
        were obtained for atom types for each atom in atom_tys (1..a) and each
        residue (1...n) in the input sequence.
    """
    n_res, n_atoms = len(atom_coords), len(atom_tys)
    coords, mask = torch.zeros(n_res, n_atoms, 3), torch.zeros(n_res, n_atoms)
    for i, res in enumerate(atom_coords):
        for atom_pos, atom_ty in enumerate(atom_tys):
            if res[atom_ty] is None:
                continue
            coords[i, atom_pos] = torch.tensor([res[atom_ty][j] for j in range(3)])
            mask[i, atom_pos] = 1
    return coords, mask.bool()
