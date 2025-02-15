import os
import numpy as np
import argparse
import h5py
from Bio.PDB import PDBParser

# Dictionary to map three-letter amino acid codes to one-letter codes.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASP": "D", "CYS": "C", "CYX": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "ASN": "N",
    "PHE": "F", "PRO": "P", "SEC": "U", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V"
}

def process_pdb_file(pdb_file):
    """
    Processes a single PDB file and returns a list of dictionaries,
    each containing for every chain:
      - chain_id: the identifier of the chain,
      - protein: the base name of the file without extension,
      - sequence: the amino acid sequence (using one-letter codes),
      - msa: a list of sequences (for now, simply the reference sequence; 
             this can be extended to include homologs from an MSA tool),
      - distance_matrix: the matrix of distances between Cα atoms,
      - target_matrix: the binary contact matrix (1 if the distance is < 8 Å).
    """
    filename = os.path.basename(pdb_file)
    base_filename, _ = os.path.splitext(filename)
    
    # Parse the structure from the PDB file.
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(base_filename, pdb_file)
    
    chains_data = []
    # Typically, PDB files have one model. We use the first model.
    model = structure[0]
    for chain in model:
        chain_id = chain.get_id()
        sequence = ""
        ca_coords = []
        
        # Iterate over residues in the chain.
        for residue in chain:
            if residue.id[0] != " ":
                continue
            
            resname = residue.get_resname().upper()
            one_letter = THREE_TO_ONE.get(resname, "X")
            
            if "CA" in residue:
                ca_atom = residue["CA"]
                ca_coords.append(ca_atom.get_coord())
                sequence += one_letter
            else:
                continue

        if not ca_coords:
            continue

        ca_coords = np.array(ca_coords)
        diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        contact_matrix = (dist_matrix < 8.0).astype(int)
        np.fill_diagonal(contact_matrix, 0)

        # For demonstration, we create an MSA as a list containing only the reference sequence.
        msa = [sequence]
        
        chain_data = {
            "chain_id": chain_id,
            "protein": base_filename,
            "sequence": sequence,
            "msa": msa,
            "distance_matrix": dist_matrix,
            "target_matrix": contact_matrix
        }
        chains_data.append(chain_data)
    
    return chains_data

def get_pdb_files_from_directory(directory):
    pdb_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdb'):
                pdb_files.append(os.path.join(root, file))
    return pdb_files

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDB files from specified directories into a single HDF5 dataset."
    )
    parser.add_argument("directories", nargs="+", help="One or more directories containing PDB files.")
    parser.add_argument("-o", "--output_dir", default="../data/processed",
                        help="Directory to save the HDF5 file (default: ../data/processed)")
    parser.add_argument("-f", "--output_file", default="pdb_dataset.h5",
                        help="Name of the output HDF5 file (default: pdb_dataset.h5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_pdb_files = []
    for directory in args.directories:
        if os.path.isdir(directory):
            pdb_files = get_pdb_files_from_directory(directory)
            all_pdb_files.extend(pdb_files)
        else:
            print(f"Warning: {directory} is not a valid directory. Skipping.")

    print(f"Found {len(all_pdb_files)} PDB files.")

    output_path = os.path.join(args.output_dir, args.output_file)
    with h5py.File(output_path, "w") as h5file:
        chains_group = h5file.create_group("chains")
        chain_index = 0
        
        for pdb_file in all_pdb_files:
            print(f"Processing file: {pdb_file}")
            chains_data = process_pdb_file(pdb_file)
            for chain_data in chains_data:
                group_name = f"chain_{chain_index:05d}"
                chain_group = chains_group.create_group(group_name)
                
                chain_group.create_dataset("distance_matrix", data=chain_data["distance_matrix"], compression="gzip")
                chain_group.create_dataset("target_matrix", data=chain_data["target_matrix"], compression="gzip")
                
                chain_group.attrs["chain_id"] = chain_data["chain_id"]
                chain_group.attrs["protein"] = chain_data["protein"]
                chain_group.attrs["sequence"] = chain_data["sequence"]
                # Save the MSA as a newline-separated string.
                chain_group.attrs["msa"] = "\n".join(chain_data["msa"])
                
                chain_index += 1

    print(f"Saved HDF5 dataset: {output_path}")

if __name__ == "__main__":
    main()