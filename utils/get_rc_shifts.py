import numpy as np
import pandas as pd

# Define random coil chemical shift reference values from Wishart et al. in J-Bio NMR, 5 (1995) 67-81
paper_order = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']

# Random coil shifts when the next residue is not a proline
rc_ala = {}
rc_ala['N'] = [123.8, 118.8, 120.4, 120.2, 120.3, 108.8, 118.2, 119.9,
               120.4, 121.8, 119.6, 118.7, np.nan, 119.8, 120.5, 115.7,
               113.6, 119.2, 121.3, 120.3]
rc_ala['H'] = [8.24, 8.32, 8.34, 8.42, 8.30, 8.33, 8.42, 8.00,
               8.29, 8.16, 8.28, 8.40, np.nan, 8.32, 8.23, 8.31, 8.15, 8.03,
               8.25, 8.12]
rc_ala['HA'] = [4.32, 4.55, 4.71, 4.64, 4.35, 4.62, 3.96, 4.73, 4.17, 4.32,
                4.34, 4.48, 4.74, 4.42, 4.34, 4.3, 4.47, 4.35, 4.12, 4.66,
                4.55]
rc_ala['C'] = [177.8, 174.6, 176.3, 176.6, 175.8, 174.9, 174.1, 176.4, 176.6,
               177.6, 176.3, 175.2, 177.3, 176.0, 176.3, 174.6, 174.7, 176.3,
               176.1, 175.9]
rc_ala['CA'] = [52.5, 58.2, 54.2, 56.6, 57.7, 45.1, 55.0, 61.1,
                56.2, 55.1, 55.4, 53.1, 63.3, 55.7, 56.0, 58.3, 61.8, 62.2,
                57.5, 57.9]
rc_ala['CB'] = [19.1, 28, 41.1, 29.9, 39.6, np.nan, 29, 38.8, 33.1,
                42.4, 32.9, 38.9, 32.1, 29.4, 30.9, 63.8, 69.8, 32.9, 29.6,
                38.8]

# When the residue in question is followed by a Proline, we instead use:
rc_pro = {}
rc_pro['N'] = [125, 119.9, 121.4, 121.7, 120.9, 109.1, 118.2, 121.7, 121.6,
               122.6, 120.7, 119.0, np.nan, 120.6, 121.3, 116.6, 116.0, 120.5,
               122.2, 120.8]
rc_pro['H'] = [8.19, 8.30, 8.31, 8.34, 8.13, 8.21, 8.37, 8.06, 8.18,
               8.14, 8.25, 8.37, np.nan, 8.29, 8.2, 8.26, 8.15, 8.02, 8.09,
               8.1]
rc_pro['HA'] = [4.62, 4.81, 4.90, 4.64, 4.9, 4.13, 5.0, 4.47, 4.60, 4.63, 4.82,
                5.0, 4.73, 4.65, 4.65, 4.78, 4.61, 4.44, 4.99, 4.84]
rc_pro['C'] = [175.9, 173, 175, 174.9, 174.4, 174.5, 172.6, 175.0, 174.8,
               175.7, 174.6, 173.6, 171.4, 174.4, 174.5, 173.1, 173.2, 174.9,
               174.8, 174.8]
rc_pro['CA'] = [50.5, 56.4, 52.2, 54.2, 55.6, 44.5, 53.3, 58.7, 54.2, 53.1,
                53.3, 51.3, 61.5, 53.7, 54.0, 56.4, 59.8, 59.8, 55.7, 55.8]
rc_pro['CB'] = [18.1, 27.1, 40.9, 29.2, 39.1, np.nan, 29.0, 38.7, 32.6, 41.7,
                32.4, 38.7, 30.9, 28.8, 30.2, 63.3, 69.8, 32.6, 28.9, 38.3]

# Create dictionaries mapping amino acid names to their random coil shifts
atom_names = ['HA', 'H', 'CA', 'CB', 'C', 'N']
randcoil_ala = {atom: dict(zip(paper_order, rc_ala[atom])) for atom in atom_names}
randcoil_pro = {atom: dict(zip(paper_order, rc_pro[atom])) for atom in atom_names}

# Dictionary for 1-letter to 3-letter amino acid code conversion
aa_1to3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
}

def get_rc_shifts(sequence: str, res_ids=None, format_1letter: bool = True) -> pd.DataFrame:
    # Convert sequence to list of 3-letter codes if needed
    if format_1letter:
        try:
            residues = [aa_1to3[aa] for aa in sequence]
        except KeyError as e:
            raise ValueError(f"Invalid amino acid code detected: {e}. Ensure all amino acids are valid.")
    else:
        # If 3-letter format, split the string every 3 characters
        residues = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        
        # Validate 3-letter codes
        for res in residues:
            if res not in paper_order:
                raise ValueError(f"Invalid amino acid code: {res}. Must be one of {paper_order}")
    
    # Use provided res_ids or generate sequential ones if not provided
    if res_ids is None:
        res_ids = list(range(1, len(residues) + 1))
    elif len(res_ids) != len(residues):
        raise ValueError(f"Length of res_ids ({len(res_ids)}) must match sequence length ({len(residues)})")
    
    # Prepare data structure for results
    results = []
    
    # Process each residue
    for i, (res, res_id) in enumerate(zip(residues, res_ids)):
        # Check if next residue is proline
        is_next_pro = (i < len(residues) - 1) and (residues[i+1] == 'PRO')
        
        # Choose appropriate reference set
        ref_set = randcoil_pro if is_next_pro else randcoil_ala
        
        # Gather shifts for this residue
        res_data = {
            'res_id': res_id,  # Use provided res_id instead of sequential
            'name': res,
            'secondary_structure': '-'  # Default value
        }
        
        # Add chemical shift values
        for atom in atom_names:
            res_data[atom] = ref_set[atom].get(res)
        
        results.append(res_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure columns are in correct order
    columns = ['res_id', 'name', 'secondary_structure'] + atom_names
    df = df[columns]
    
    return df

if __name__ == "__main__":
    sequence = "MALWMRLLPLLALLALWGPDPAAA"
    df = get_rc_shifts(sequence, format_1letter=True)
    print(df)
