"""
This script processes a folder of PDB (Protein Data Bank) files and converts
them into an HDF5 dataset containing sequence, structural, and contact
information for protein chains.

Steps:
1. Extract chain sequence and structural information:
   - Reads PDB files and extracts sequence, distance matrix, contact matrix,
     and torsion angles for each chain.
   - Converts 3-letter amino acid codes to 1-letter codes.

2. Compute Multiple Sequence Alignments (MSA):
   - Uses MAFFT to align sequences within the same PDB file.
   - Outputs aligned sequences for each chain.

3. Save results in an HDF5 file:
   - Stores extracted features (sequence, MSA, distance matrix, contact matrix,
     and angle matrix) in an HDF5 file.

Arguments:
- `--pdb_folder` (str, required): Folder containing input PDB files.
- `--output_dataset` (str, required): Output HDF5 file path.
- `--tmp_dir` (str, default="tmp"): Temporary directory for intermediate files.
- `--mafft_path` (str, default="mafft"): Path to the MAFFT executable.
- `--contact_threshold` (float, default=8.0): Distance threshold (Å) for
  determining residue-residue contacts.

Outputs:
- An HDF5 dataset with per-chain groups storing sequence and structural data.

Notes:
- The script relies on Biopython for PDB parsing and sequence processing.
- The MAFFT alignment step is skipped if only a single chain is present.
- The output HDF5 file is compressed for efficient storage.
"""

import argparse
import os
import subprocess

import h5py
import numpy as np
from Bio import AlignIO
from Bio.PDB import PDBParser, PPBuilder
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASP": "D",
    "CYS": "C",
    "CYX": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HIE": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "ASN": "N",
    "PHE": "F",
    "PRO": "P",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def extract_chain_info(pdb_path, chain_id, contact_threshold=8.0):
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)

    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return None

    model = structure[0]
    chain_obj = None
    seq = ""
    coords = []
    for chain in model:
        if chain.get_id() == chain_id:
            chain_obj = chain
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resname = residue.get_resname().upper()
                aa = THREE_TO_ONE.get(resname, "X")
                if "CA" in residue:
                    seq += aa
                    coords.append(residue["CA"].get_coord())
            break
    if chain_obj is None or len(seq) == 0:
        return None
    L = len(seq)
    coords = np.array(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    contact_matrix = (distance_matrix < contact_threshold).astype(np.float32)
    np.fill_diagonal(contact_matrix, 0)

    ppb = PPBuilder()
    angle_list = [(0.0, 0.0)] * L
    for pp in ppb.build_peptides(chain_obj):
        phi_psi = pp.get_phi_psi_list()
        for i, (phi, psi) in enumerate(phi_psi):
            if i < L:
                angle_list[i] = (
                    phi if phi is not None else 0.0,
                    psi if psi is not None else 0.0,
                )
    angle_matrix = np.array(angle_list, dtype=np.float32)
    return {
        "sequence": seq,
        "distance_matrix": distance_matrix,
        "contact_matrix": contact_matrix,
        "angle_matrix": angle_matrix,
    }


def compute_msa_for_pdb(pdb_path, chains, tmp_dir, mafft_path="mafft"):
    fasta_path = os.path.join(tmp_dir, "temp.fasta")
    records = []
    seq_dict = {}
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return {}
    model = structure[0]
    for chain in model:
        cid = chain.get_id()
        if cid in chains:
            seq = ""
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resname = residue.get_resname().upper()
                aa = THREE_TO_ONE.get(resname, "X")
                if "CA" in residue:
                    seq += aa
            if seq:
                seq_dict[cid] = seq
                records.append(SeqRecord(Seq(seq), id=cid, description=""))
    from Bio import SeqIO

    SeqIO.write(records, fasta_path, "fasta")
    aligned_fasta = os.path.join(tmp_dir, "aligned.fasta")
    if len(records) > 1:
        cmd = [mafft_path, "--quiet", "--auto", fasta_path]
        try:
            with open(aligned_fasta, "w") as out:
                subprocess.run(cmd, stdout=out, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running MAFFT on {fasta_path}: {e}")
            return {}
        alignment = AlignIO.read(aligned_fasta, "fasta")
        aligned_dict = {record.id: str(record.seq) for record in alignment}
        os.remove(fasta_path)
        os.remove(aligned_fasta)
    else:
        aligned_dict = {list(seq_dict.keys())[0]: list(seq_dict.values())[0]}
        os.remove(fasta_path)
    msa_list_dict = {}
    for query in chains:
        if query not in aligned_dict:
            continue
        query_aligned = aligned_dict[query]
        msa_list = [query_aligned] + [
            aligned_dict[c] for c in aligned_dict if c != query
        ]
        msa_list_dict[query] = msa_list
    return msa_list_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of PDB files to an HDF5 dataset."
    )
    parser.add_argument(
        "--pdb_folder", required=True, help="Folder containing PDB files"
    )
    parser.add_argument("--output_dataset", required=True, help="Output HDF5 file")
    parser.add_argument(
        "--tmp_dir", default="tmp", help="Temporary directory for intermediate files"
    )
    parser.add_argument(
        "--mafft_path", default="mafft", help="Path to the MAFFT executable"
    )
    parser.add_argument(
        "--contact_threshold",
        type=float,
        default=8.0,
        help="Distance threshold for contact (Å)",
    )
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    out_h5 = h5py.File(args.output_dataset, "w")
    chains_group = out_h5.create_group("chains")
    chain_index = 0

    pdb_files = [f for f in os.listdir(args.pdb_folder) if f.lower().endswith(".pdb")]
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        pdb_path = os.path.join(args.pdb_folder, pdb_file)
        parser_pdb = PDBParser(QUIET=True)
        try:
            structure = parser_pdb.get_structure(pdb_file, pdb_path)
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            continue
        model = structure[0]
        chain_ids = [
            chain.get_id()
            for chain in model
            if any("CA" in residue for residue in chain)
        ]
        if not chain_ids:
            continue

        aligned_dict = compute_msa_for_pdb(
            pdb_path, chain_ids, args.tmp_dir, mafft_path=args.mafft_path
        )

        for cid in chain_ids:
            info = extract_chain_info(
                pdb_path, cid, contact_threshold=args.contact_threshold
            )
            if info is None:
                continue
            msa_seq = aligned_dict.get(cid, info["sequence"])
            group_name = f"chain_{chain_index:05d}"
            grp = chains_group.create_group(group_name)
            grp.attrs["pdb_file"] = pdb_file
            grp.attrs["chain_id"] = cid
            grp.attrs["sequence"] = info["sequence"]
            grp.attrs["msa"] = msa_seq
            grp.create_dataset(
                "contact_matrix", data=info["contact_matrix"], compression="gzip"
            )
            grp.create_dataset(
                "distance_matrix", data=info["distance_matrix"], compression="gzip"
            )
            grp.create_dataset(
                "angle_matrix", data=info["angle_matrix"], compression="gzip"
            )
            chain_index += 1

    out_h5.close()
    print(f"Full HDF5 dataset saved to {args.output_dataset}")


if __name__ == "__main__":
    main()
