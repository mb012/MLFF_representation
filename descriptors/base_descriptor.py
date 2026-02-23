from collections import defaultdict
from data.external.spartap_features import (PDB_SPARTAp_DataReader)

from abc import ABC, abstractmethod
import pickle
import torch
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from ase import Atoms
from tqdm import tqdm
from utils.process_data import load_struct, struct_to_dataframe
import concurrent.futures
import os
from pathlib import Path
import numpy as np
import multiprocessing


class BaseDescriptor(ABC):
    def __init__(self, desc_name, bmrb_pdb_path="./data/bmrb-pdb-refs.csv", valid_seq_path="./data/valid-exp-sequences.csv"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bmrb_pdf_file = bmrb_pdb_path
        self.valid_seq_path = valid_seq_path
        
        self.desc_path = "./data/descriptors_dataset/"+desc_name
        os.makedirs(self.desc_path, exist_ok=True)        

        
    def get_envs(self, data_folder:str, rmax=5.0, num_workers=10):
        entries = self._get_entries()
        
        multiprocessing.set_start_method('spawn', force=True)
        envs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_entry, entry._asdict(), data_folder, rmax) for entry in entries]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing envs"):
                envs.append(future.result())
        # list of env dictionaries and features
        return envs
    
    def _get_entries(self):
        valid_exp_sequences = pd.read_csv(self.valid_seq_path).drop(["Unnamed: 0"], axis=1).set_index("bmrb_id")
        bmrb_pdb_mapping = pd.read_csv(self.bmrb_pdf_file).drop(["Unnamed: 0"], axis=1).set_index("bmrb_id")
        valid_exp_sequences = valid_exp_sequences.join(bmrb_pdb_mapping, how="inner")
        print(f"envs num: {len(list(valid_exp_sequences.itertuples()))}")
        return list(valid_exp_sequences.itertuples())
        
    def _process_entry(self, entry, data_folder:str, rmax:float=5.0):
        bmrb_id, pdb_id = entry['Index'], entry['pdb_id']
        print(f"Processing PDB ID: {pdb_id}")
        folder_name = f"{bmrb_id}_{pdb_id}"
        folder_path = Path(data_folder) / folder_name
        af_struct_filename = folder_path / f"af2_00_relaxed.pdb"

        if not af_struct_filename.exists():
            print(f"missing pdb id {pdb_id}")
            return
        
        structure = load_struct(af_struct_filename)
        structure_df = struct_to_dataframe(structure)
        
        structure_df["pdb_id"] = pdb_id
        structure_df["bmrb_id"] = bmrb_id
        structure_df["af_file_name"] = af_struct_filename
        environments = self._extract_amino_acid_env(structure_df, rmax=rmax)
        features = PDB_SPARTAp_DataReader().df_from_file_3res(str(af_struct_filename.absolute()))
        return environments, features
        
    def _extract_amino_acid_env(self, df, rmax=5.0):
        """
        Extract environments per amino acid based on proximity (rmax)
        """
        X = np.vstack(df['coords'].values)
        neight = NearestNeighbors().fit(X)
        environments = {}

        for res_id, sub_df in df.groupby('res_id'):
            Y = np.vstack(sub_df['coords'].values)
            indices = neight.radius_neighbors(Y, radius=rmax, return_distance=False, sort_results=False)
            
            # Collect unique residues from neighbors
            res_indices = {df.iloc[idx]['res_id'] for idx in set(np.concatenate(indices))}
            environments[res_id] = df[df['res_id'].isin(res_indices)]
            
        return environments
    
    @abstractmethod 
    def get_model(self, model:str):
        pass
    
    def _compute_atoms(self, env):
        env_coords = np.stack([env["coords"].iloc[i, ] for i in range(env.shape[0])])
        element_ids = env["element"].values
        atoms_obj_env = Atoms(symbols=element_ids, positions=env_coords)
        return atoms_obj_env
    
    def _get_atom_descriptor(self, descriptors, res_id, env):
        env = env.reset_index(drop=True)
        assert descriptors.shape[0] == env.shape[0]
        indices = env.index[env["res_id"] == res_id].tolist()
        atoms = env.loc[indices, "atom"].tolist()
        result = {atoms[i].rstrip('1'): descriptors[idx] for i, idx in enumerate(indices)}
        return result

    
    @abstractmethod    
    def _calc_descriptors(self, model, atoms_obj_env, **kwargs):
        pass
    
    def _filter_exisiting_files(self, envs):
        filtered_envs = []
        for amino_acid_env, features in envs:
            # it's enough to check only the first env as the file is created only if processed all the atoms in the amino acid
            env = next(iter(amino_acid_env.items()))[1]
            bmrb_id, pdb_id = env["bmrb_id"].unique()[0], env["pdb_id"].unique()[0]
            filename = f"{bmrb_id}_{pdb_id}"
            for folder in os.listdir(self.desc_path):
                if not Path(f"{self.desc_path}/{folder}/{filename}.pkl").exists():
                    filtered_envs += [(amino_acid_env, features)]
                    break
        return filtered_envs
    
    def _get_empty_targets_dict(self, atoms):
        empty_targets = {"name": np.nan, "secondary_structure":np.nan, "cs":{}, "rc_cs":{}, "pka": np.nan}
        if atoms is not None:
            for a in atoms:
                empty_targets["cs"][a] = np.nan
                empty_targets["rc_cs"][a] = np.nan
        return empty_targets
    
    def _get_targets(self, path, res_id, features, atoms):
        df_target = pd.read_csv(path)
        assert len(df_target) > 0
        targets=self._get_empty_targets_dict(atoms)
        res_df = df_target[df_target["res_id"]==res_id]
        if len(res_df) > 0:
            targets["name"] = res_df["name"].values[0]
            ss = res_df["secondary_structure"].values[0]
            targets["secondary_structure"] = ss if ss != "-" else np.nan
            pka = res_df["pKa"].values[0]
            targets["pka"] = pka if pka != "-" else np.nan
            if atoms is not None:
                for atom in atoms:
                    cs = res_df[atom].values[0] if atom in res_df.columns else "-"
                    targets["cs"][atom] = cs if cs != "-" else np.nan
                    res_features = features.query("RES_NUM==@res_id")
                    if f"RC_{atom}" in res_features.columns:
                        targets["rc_cs"][atom] = res_features[f"RC_{atom}"].values[0]
                    elif f"rc_{atom}" in res_df.columns:
                        targets["rc_cs"][atom] = res_df[f"rc_{atom}"].values[0]
                    else:
                        targets["rc_cs"][atom] = np.nan
        return targets

    def generate_descriptors(self, data_folder:str, model_name:str, rmax:float=5.0, 
                             num_workers:int=20, filter_exisiting_files=True, **kwargs):
        envs = self.get_envs(data_folder, rmax=rmax, num_workers=num_workers)
        envs = self._filter_exisiting_files(envs) if filter_exisiting_files else envs
        model = self.get_model(model_name)
        
        for amino_acid_env, features in tqdm(envs, desc="Computing descriptors"):
            prot_atom_descriptors = defaultdict(list)
            for i, (res_id, env) in enumerate(amino_acid_env.items()):
                atoms_obj_env = self._compute_atoms(env)
                descriptors = self._calc_descriptors(model, atoms_obj_env, **kwargs)
                descriptor_dict = self._get_atom_descriptor(descriptors, res_id, env)
                bmrb_id, pdb_id = env["bmrb_id"].unique()[0], env["pdb_id"].unique()[0]
                targets = self._get_targets(f"./data/target_data/{pdb_id.upper()}_{bmrb_id}.csv", res_id, features, descriptor_dict.keys())
                for atom in descriptor_dict.keys():
                    coords = env[(env["res_id"] == res_id) & (env["atom"] == atom)]["coords"]
                    prot_atom_descriptors[atom].append(
                    {
                        "bmrb_id": bmrb_id,
                        "pdb_id": pdb_id,
                        "descriptor": descriptor_dict[atom],
                        "name": targets["name"],
                        "secondary_structure": targets["secondary_structure"],
                        "cs": targets["cs"][atom],
                        "rc_cs": targets["rc_cs"][atom],
                        "res_id": res_id,
                        "atom": atom,
                        "pka": targets["pka"],
                        "coord": coords.values[0] if atom in env[env["res_id"]==res_id]["atom"].tolist() else np.nan
                    })
                    
            for atom in prot_atom_descriptors.keys():
                os.makedirs(Path(self.desc_path) / atom, exist_ok=True)        
                filename = f"{bmrb_id}_{pdb_id}"
                with open(f"{self.desc_path}/{atom}/{filename}.pkl", "wb") as f:
                    pickle.dump(prot_atom_descriptors[atom], f)
                    f.close()