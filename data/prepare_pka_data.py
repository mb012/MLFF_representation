import os
import torch
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
# TODO
# Use CA descriptors to predict the pKa values of GLU residue. Take all CA mace descriptors 
# And then filter out those that are not in GLU?
# That should give a target value
aa_map = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
          'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
          'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
          'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

pka_dirs = ["mace_invariant_pka_tyr", "aimnet_pka_tyr", "orb_pka_tyr"]
mace_roots = ["mace_invariant", "aimnet", "orb"]

# atom = "CA"
atoms = ["N", "CA", "C", "H", "HA", "CB"]
pka_root = "pka_results"


mace_embeddings = []
pkA_values = []
residue_names = []
state_ids = []
for k in range(len(pka_dirs)):
    total = 0
    pka_dir, mace_root = pka_dirs[k], mace_roots[k]
    for atom in atoms:
        os.makedirs(os.path.join(pka_dir, atom), exist_ok=True)
        for entry in tqdm(os.listdir(pka_root)):
            bmrb_id, pdb_id = entry.split("_")

            protonation_entry_df = pd.read_csv(os.path.join(pka_root, entry, "protonation_states.csv"))
            protonation_entry_df['pK'] = protonation_entry_df['pK'].replace('-----', np.nan)

            filtered_df = protonation_entry_df[protonation_entry_df['res_name'] == "TYR"]
            filtered_df_pka, filtered_df_res_num = filtered_df["pK"], np.array(filtered_df["res_number"])

            descriptor_file = os.path.join(mace_root, atom, f"{bmrb_id}_{pdb_id}.pkl")
            with open(descriptor_file, "rb") as file:
                data = pickle.load(file)

            # Take descriptors that are in res_number
            new_descriptor = []
            for entry_i in data:
                entry_copy = entry_i.copy()
                if entry_copy["res_id"] in filtered_df_res_num:
                    # Check that it's a GLU
                    df_i = filtered_df[filtered_df.res_number == entry_copy["res_id"]]
                    pka_val = float(df_i['pK'].iloc[0])

                    if np.isnan(pka_val):
                        continue

                    entry_copy["pka"] = pka_val
                    pkA_values.append(pka_val)

                    if not(isinstance(entry_copy["name"], float) and np.isnan(entry_copy["name"])):
                        assert entry_copy["name"] == aa_map[df_i['res_name'].iloc[0]]
                    new_descriptor.append(entry_copy)

            write_file_path = os.path.join(pka_dir, atom, f"{bmrb_id}_{pdb_id}.pkl")
            if len(new_descriptor) > 0:
                with open(write_file_path, "wb") as file:
                    pickle.dump(new_descriptor, file)

                total += len(new_descriptor)
    print(f"{mace_root}: {total}")