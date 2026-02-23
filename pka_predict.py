from modulefinder import packagePathMap
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from ase import Atoms
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from mace.calculators import mace_off
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import GCNConv
from torch.optim.swa_utils import AveragedModel
from utils.get_rc_shifts import get_rc_shifts
from utils.process_data import load_struct, struct_to_dataframe
# from get_rc_shifts import get_rc_shifts
import pickle


MACE_DESCRIPTOR_MODEL = "large"
MACE_PKA_MODEL_PATHS = {
    # "ASP": TODO,
    # "GLU": TODO,
    # "LYS": TODO,
    # "HIS": TODO,
}

aa_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'SEC': 'U', 'PYL': 'O',
    'MSE': "M"
}

inv_aa_map = {v: k for k, v in aa_map.items()}

def parse_model_paths(items):
    d = {}
    for item in items:
        try:
            key, value = item.split("=", 1)
        except ValueError:
            raise ValueError(f"Invalid format '{item}', expected RES=PATH")
        d[key.upper()] = value
    return d

# GNN model for PkA prediction
class ImprovedGNN(torch.nn.Module):
    def __init__(self, in_dim, dim, out_dim, num_layers=10, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()  
        self.convs.append(GCNConv(in_dim, dim))
        self.norms.append(LayerNorm(dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(dim, dim))
            self.norms.append(LayerNorm(dim))
        
        self.convs.append(GCNConv(dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, coord):
        for i, (conv, norm) in enumerate(zip(self.convs[:-1], self.norms)):
            x = conv(x, edge_index)
            x = norm(x)  
            x = F.silu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x[-1]

class PkaPredictor:
    def __init__(self, mace_model_path=None, pka_model_paths=None, atoms=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.atoms = atoms
        print(f"Using device: {self.device}")
        
        # Use predefined MACE model if not specified
        if mace_model_path is None:
            mace_model_path = MACE_DESCRIPTOR_MODEL
        
        # Load MACE model for descriptors
        self.descriptors_model = mace_off(model=mace_model_path, device=device, default_dtype="float32")
        self.descriptors_model.models[0].eval()
        
        # Use predefined PKA models if not specified
        if pka_model_paths is None:
            pka_model_paths = MACE_PKA_MODEL_PATHS
        else:
            pka_model_paths = parse_model_paths(pka_model_paths)


        # Load PKA prediction model
        self.pka_models = {}
        self.amino_acid_types = []
        self.atom_types = atoms[:]

        for amino_acid_type in pka_model_paths.keys():
            self.amino_acid_types.append(amino_acid_type)

            state_dict = torch.load(pka_model_paths[amino_acid_type], map_location=self.device)["ema_model"]
            pka_model = AveragedModel(ImprovedGNN(448, 128, 1, 10, 0.3))
            pka_model.load_state_dict(state_dict)
            pka_model.eval()
            self.pka_models[amino_acid_type] = pka_model.to(self.device)
        print(f"Loaded pKa models for amino acids: {', '.join(self.amino_acid_types)}")
    # ({'dim': 128, 'num_layers': 10, 'dropout': 0.3}, 448, 1)
    def extract_environments(self, structure_df, rmax=5.0, save_dir=None, prefix=''):
        """Extract environments per amino acid based on proximity (rmax)"""
        X = np.vstack(structure_df['coords'].values)
        neight = NearestNeighbors().fit(X)
        environments = {}

        for res_id, sub_df in structure_df.groupby('res_id'):
            Y = np.vstack(sub_df['coords'].values)
            indices = neight.radius_neighbors(Y, radius=rmax, return_distance=False, sort_results=False)
            
            # Collect unique residues from neighbors
            res_indices = {structure_df.iloc[idx]['res_id'] for idx in set(np.concatenate(indices))}
            environments[res_id] = structure_df[structure_df['res_id'].isin(res_indices)]
            environments[res_id].reset_index(drop=True, inplace=True)
        
        # Save environments if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            env_path = os.path.join(save_dir, f'{prefix}environments.pkl')
            with open(env_path, 'wb') as f:
                pickle.dump(environments, f)
            print(f"Saved environments to {env_path}")
            
        return environments
    
    def compute_atoms(self, env):
        """Create ASE Atoms object from environment dataframe"""
        env_coords = np.stack([env["coords"].iloc[i, ] for i in range(env.shape[0])])
        element_ids = env["element"].values
        atoms_obj_env = Atoms(symbols=element_ids, positions=env_coords)
        return atoms_obj_env
    
    def get_descriptors(self, environments, save_dir=None, prefix=''):
        """Compute MACE descriptors for each environment"""
        descriptors_dict = {}
        order_1_descriptors_dict = {}
        
        for res_id, env in tqdm(environments.items(), desc="Computing descriptors"):
            amino_acid = inv_aa_map[env[env["res_id"] == res_id]["name"].unique()[0]]
            if amino_acid not in self.amino_acid_types:
                continue
            atoms_obj_env = self.compute_atoms(env)
            descriptors = self.descriptors_model.get_descriptors(
                atoms_obj_env, 
                invariants_only=True, 
                num_layers=-1
            )
            order_1_descriptors = self.descriptors_model.get_descriptors(
                atoms_obj_env, 
                invariants_only=False, 
                num_layers=-1
            )
            descriptors = torch.tensor(descriptors, dtype=torch.float32, device=self.device)
            order_1_descriptors = torch.tensor(order_1_descriptors, dtype=torch.float32, device=self.device)
            # import ipdb; ipdb.set_trace()
            # Extract descriptors for each atom in the residue
            atom_descriptors = {}
            order_1_atom_descriptors = {}
            for atom_type in self.atom_types:
                indices = env[(env["res_id"] == res_id) & (env["atom"] == atom_type)].index.tolist()
                if indices:
                    atom_idx = indices[0] - env.index[0]  # Adjust index to descriptor array
                    if atom_idx < len(descriptors):
                        atom_descriptors[atom_type] = descriptors[atom_idx]
                    if atom_idx < len(order_1_descriptors):
                        order_1_atom_descriptors[atom_type] = order_1_descriptors[atom_idx]
            
            atom_descriptors["all"] = torch.stack(list(atom_descriptors.values()))
            order_1_atom_descriptors["all"] = torch.stack(list(order_1_atom_descriptors.values()))
            descriptors_dict[(res_id, amino_acid)] = atom_descriptors
            order_1_descriptors_dict[(res_id, amino_acid)] = order_1_atom_descriptors
        # Save descriptors if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Convert tensors to numpy for storage
            save_dict = {}
            save_dict_order_1 = {}
            for res_id, atom_dict in descriptors_dict.items():
                residue_id, amino_acid = res_id
                save_dict[residue_id] = {}
                save_dict_order_1[residue_id] = {}
                for atom_type, descriptor in atom_dict.items():
                    if isinstance(descriptor, torch.Tensor):
                        save_dict[residue_id][atom_type] = descriptor.cpu().numpy()
                        save_dict_order_1[residue_id][atom_type] = order_1_descriptors_dict[res_id][atom_type].cpu().numpy()
                    else:
                        save_dict[residue_id][atom_type] = descriptor
                        save_dict_order_1[residue_id][atom_type] = order_1_descriptors_dict[residue_id][atom_type]
                        
            desc_path = os.path.join(save_dir, f'{prefix}descriptors.pkl')
            with open(desc_path, 'wb') as f:
                pickle.dump(save_dict, f)
            print(f"Saved descriptors to {desc_path}")
            
            desc_path_order_1 = os.path.join(save_dir, f'{prefix}descriptors_order_1.pkl')
            with open(desc_path_order_1, 'wb') as f:
                pickle.dump(save_dict_order_1, f)
            print(f"Saved descriptors to {desc_path_order_1}")
            
        return descriptors_dict
    
    def predict_pka(self, descriptors_dict):
        """Predict pKa using the descriptors"""
        predictions = defaultdict(dict)
        
        for res_id, atom_descriptors in descriptors_dict.items():
            residue_id, amino_acid = res_id
            pka_model_aa = self.pka_models[amino_acid]

            descriptor_tensor = atom_descriptors["all"]
            num_nodes = descriptor_tensor.shape[0]
            row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
            edges = torch.vstack([row.flatten(), col.flatten()]).to(device=self.device)
            with torch.no_grad():
                pred_pka = pka_model_aa(descriptor_tensor, edges, None).item()
            predictions[res_id] = pred_pka
        return predictions
    
    def extract_sequence(self, structure_df):
        """Extract protein sequence from structure dataframe"""
        # Get unique residues sorted by residue ID
        residues = structure_df[['res_id', 'name']].drop_duplicates().sort_values('res_id')
        
        # Convert 3-letter codes to 1-letter and join
        sequence = ''.join([name for name in residues['name']])
        return sequence, list(residues['res_id'])
    
    def predict_from_file(self, file_path, rmax=5.0, output_csv=None, save_dir=None, prefix=''):
        """End-to-end prediction from PDB/CIF file"""
        # Create save directory based on input file if not specified
        if save_dir is None and output_csv:
            save_dir = os.path.dirname(output_csv)
        elif save_dir is None:
            # Use a subfolder in the current directory based on the input filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_dir = f"case_study_{base_name}"
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Using directory for saving results: {save_dir}")
        
        # Load structure
        structure = load_struct(file_path)
        structure_df = struct_to_dataframe(structure)

        # Extract sequence for random coil shifts
        sequence, res_ids = self.extract_sequence(structure_df)
        print(f"Extracted sequence: {sequence}")
        
        # Extract environments
        print("Extracting environments...")
        environments = self.extract_environments(structure_df, rmax, save_dir, prefix)
        
        # Compute descriptors
        print("Computing MACE descriptors...")
        descriptors_dict = self.get_descriptors(environments, save_dir, prefix)
        
        # Predict chemical shifts
        print("Predicting pKa...")
        predictions = self.predict_pka(descriptors_dict)
        
        # Convert to DataFrame
        results = []
        for res_id, pka in predictions.items():
            residue_id, amino_acid = res_id

            results.append({
                "res_id": residue_id,
                "name": amino_acid,
                "pka": pka,
            })
        
        results_df = pd.DataFrame(results)
        
        # Save results if requested
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description='Predict chemical shifts from a PDB/CIF file')
    parser.add_argument('--pdb', type=str, default=None, required=False, help='Path to PDB/CIF file')
    parser.add_argument('--mace', type=str, default=None, help='Path to MACE model (defaults to MACE_DESCRIPTOR_MODEL)')
    parser.add_argument('--pka_model_paths', type=str, nargs='+', default=MACE_PKA_MODEL_PATHS, metavar="RES=PATH", help='Paths to pKa prediction models (defaults to MACE_PKA_MODEL_PATHS), e.g. LYS=path HIS=path')
    parser.add_argument('--atoms', type=str, nargs='+', default=["N", "CA", "C", "H", "HA", "CB"], help='Atom descriptors used to predict pKa')
    parser.add_argument('--rmax', type=float, default=5.0, help='Maximum radius for environment extraction')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save intermediate results')
    parser.add_argument('--prefix', type=str, default='', help='Optional prefix for saved files')
    
    args = parser.parse_args()
    
    predictor = PkaPredictor(
        mace_model_path=args.mace,
        pka_model_paths=args.pka_model_paths,
        device=args.device,
        atoms=args.atoms
    )
    
    results = predictor.predict_from_file(
        file_path=args.pdb,
        rmax=args.rmax,
        output_csv=args.output,
        save_dir=args.save_dir,
        prefix=args.prefix
    )
    
    print("Prediction completed!")
    print(results.head())

if __name__ == "__main__":
    main()


