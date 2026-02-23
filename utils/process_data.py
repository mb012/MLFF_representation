from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data import IUPACData
import pandas as pd
import os

ACIDS_1TO3 = {
    a.upper(): aaa.upper() for a, aaa in IUPACData.protein_letters_1to3.items()
}
ACIDS_3TO1 = {aaa: a for a, aaa in ACIDS_1TO3.items()}


def load_struct(file, structid='mystruct'):
    # Determine file type by extension
    file_ext = os.path.splitext(file)[1].lower()
    
    if file_ext == '.cif':
        parser = MMCIFParser()
    else:  # Default to PDB parser for .pdb and other formats
        parser = PDBParser()
        
    struct = parser.get_structure(structid, file)
    return struct


def struct_to_dataframe(struct):
    model = [*struct.get_models()][0]
    chains = [*model.get_chains()]
    assert len(chains) == 1
    chain = chains[0]

    records = []
    for res in chain.get_residues():
        # Handle potential difference in residue naming between PDB and CIF files
        resname = res.get_resname()
        # Skip non-standard residues or handle special cases
        if resname not in ACIDS_3TO1:
            continue
            
        for atom in res.get_atoms():
            record = {
                'res_id': res.id[1],
                'name': ACIDS_3TO1[resname],
                'atom': atom.name,
                'element': atom.name[0],
                'bfactor': atom.get_bfactor(),
                'coords': tuple(atom.get_coord()),
            }
            records.append(record)
    return pd.DataFrame(records)
