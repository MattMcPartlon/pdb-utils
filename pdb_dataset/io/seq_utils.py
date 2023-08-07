from typing import List, Any, Union

import numpy as np
import os
from Bio import SeqIO


def load_fasta_file(seq_file, returnStr=True) -> Union[str, List]:
    """Load a fasta file.

    :param seq_file: file to read (fasta) sequence from.
    :param returnStr: whether to return string representation (default) or list.
    :return: sequence as string or list.
    """
    if not os.path.isfile(seq_file) or not seq_file.endswith(".fasta"):
        # pylint: disable-next=broad-exception-raised
        raise Exception("ERROR: an invalid sequence file: ", seq_file)
    record = SeqIO.read(seq_file, "fasta")
    return str(record.seq) if returnStr else record.seq


def load_npy(path) -> Any:
    """Load .npy data type"""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e1:  # pylint: disable=broad-exception-caught
        try:
            data = np.load(path, allow_pickle=True, encoding="latin1")
        except Exception as e2:
            print(
                f"could not load file {path}\n"
                f" exception when loading with default encoding {e1}"
                f"\n exception when loading with latin1 {e2}"
            )
            raise e2
    try:
        data = data.item()
    except:  # pylint: disable=bare-except, # noqa
        pass  # noqa
    return data
