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
import pickle


MACE_DESCRIPTOR_MODEL = "large"
# MACE_CS_MODELS = {
#     "CA": # TODO add path,
#     "CB": # TODO add path,
#     "HA": # TODO add path,
#     "N": # TODO add path,
#     "H": # TODO add path,
#     "C": # TODO add path,
# }



# GNN model for CS prediction
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
        return x

class CSPredictor:
    def __init__(self, mace_model_path=None, cs_model_paths=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Use predefined MACE model if not specified
        if mace_model_path is None:
            mace_model_path = MACE_DESCRIPTOR_MODEL
        
        # Load MACE model for descriptors
        self.descriptors_model = mace_off(model=mace_model_path, device=device, default_dtype="float32")
        self.descriptors_model.models[0].eval()
        
        # Use predefined CS models if not specified
        if cs_model_paths is None:
            cs_model_paths = list(MACE_CS_MODELS.values())
        
        # Load CS prediction models
        self.cs_models = {}
        self.atom_types = []
        for path in cs_model_paths:
            atom_type = Path(path).stem.split('_')[-3]  # Extract atom type from filename
            self.atom_types.append(atom_type)
            state_dict = torch.load(path, map_location=self.device)["ema_model"]
            cs_model = AveragedModel(ImprovedGNN(448, 256, 1, 5, 0.3))
            cs_model.load_state_dict(state_dict)
            cs_model.eval()
            self.cs_models[atom_type] = cs_model.to(self.device)
        
        print(f"Loaded CS models for atoms: {', '.join(self.atom_types)}")
    
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
            
            descriptors_dict[res_id] = atom_descriptors
            order_1_descriptors_dict[res_id] = order_1_atom_descriptors
        # Save descriptors if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Convert tensors to numpy for storage
            save_dict = {}
            save_dict_order_1 = {}
            for res_id, atom_dict in descriptors_dict.items():
                save_dict[res_id] = {}
                save_dict_order_1[res_id] = {}
                for atom_type, descriptor in atom_dict.items():
                    if isinstance(descriptor, torch.Tensor):
                        save_dict[res_id][atom_type] = descriptor.cpu().numpy()
                        save_dict_order_1[res_id][atom_type] = order_1_descriptors_dict[res_id][atom_type].cpu().numpy()
                    else:
                        save_dict[res_id][atom_type] = descriptor
                        save_dict_order_1[res_id][atom_type] = order_1_descriptors_dict[res_id][atom_type]
                        
            desc_path = os.path.join(save_dir, f'{prefix}descriptors.pkl')
            with open(desc_path, 'wb') as f:
                pickle.dump(save_dict, f)
            print(f"Saved descriptors to {desc_path}")
            
            desc_path_order_1 = os.path.join(save_dir, f'{prefix}descriptors_order_1.pkl')
            with open(desc_path_order_1, 'wb') as f:
                pickle.dump(save_dict_order_1, f)
            print(f"Saved descriptors to {desc_path_order_1}")
            
        return descriptors_dict
    
    def predict_cs(self, descriptors_dict):
        """Predict chemical shifts using the descriptors"""
        predictions = defaultdict(dict)
        
        for res_id, atom_descriptors in descriptors_dict.items():
            for atom_type, descriptor in atom_descriptors.items():
                if atom_type in self.cs_models:
                    # Prepare input for CS model
                    descriptor_tensor = descriptor.unsqueeze(0)  # Add batch dimension
                    edges = torch.tensor([[0], [0]], device=self.device)  # Self-connection
                    
                    # Predict chemical shift
                    with torch.no_grad():
                        pred_cs = self.cs_models[atom_type](descriptor_tensor, edges, None).item()
                    
                    predictions[res_id][atom_type] = pred_cs
        
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
        
        # Get random coil shifts
        print("Computing random coil shifts...")
        rc_shifts = get_rc_shifts(sequence, res_ids=res_ids, format_1letter=True)
        
        # Extract environments
        print("Extracting environments...")
        environments = self.extract_environments(structure_df, rmax, save_dir, prefix)
        
        # Compute descriptors
        print("Computing MACE descriptors...")
        descriptors_dict = self.get_descriptors(environments, save_dir, prefix)
        
        # Predict chemical shifts
        print("Predicting chemical shifts...")
        predictions = self.predict_cs(descriptors_dict)
        
        # Convert to DataFrame
        results = []
        for res_id, atom_shifts in predictions.items():
            res_name = structure_df[structure_df['res_id'] == res_id]['name'].iloc[0]
            for atom_type, shift in atom_shifts.items():
                # Find corresponding random coil shift
                rc_value = None
                rc_row = rc_shifts[rc_shifts['res_id'] == res_id]
                if not rc_row.empty and atom_type in rc_row.columns:
                    rc_value = rc_row[atom_type].iloc[0]
                
                results.append({
                    'res_id': res_id,
                    'name': res_name,
                    'atom': atom_type,
                    'rel_cs': shift,
                    'rc_cs': rc_value,
                    'predicted_cs': None if rc_value is None else shift + rc_value
                })
        
        results_df = pd.DataFrame(results)
        
        # Save results if requested
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
            
        # Plot results
        self.plot_results(results_df)
        
        return results_df
    
    def plot_results(self, results_df):
        """Plot predicted chemical shifts by atom type"""
        for atom_type in self.atom_types:
            atom_data = results_df[results_df['atom'] == atom_type]
            if not atom_data.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(atom_data['res_id'], atom_data['predicted_cs'], 'o-', label=f'Predicted {atom_type}')
                
                # Plot random coil shifts if available
                if 'rc_cs' in atom_data.columns and not atom_data['rc_cs'].isna().all():
                    plt.plot(atom_data['res_id'], atom_data['rc_cs'], 's--', label=f'Random Coil {atom_type}')
                
                # Plot secondary shifts if available
                if 'secondary_cs' in atom_data.columns and not atom_data['secondary_cs'].isna().all():
                    plt.figure(figsize=(10, 6))
                    plt.plot(atom_data['res_id'], atom_data['secondary_cs'], 'o-', color='green')
                    plt.xlabel('Residue ID')
                    plt.ylabel(f'{atom_type} Secondary Shift (ppm)')
                    plt.title(f'{atom_type} Secondary Chemical Shifts (Predicted - Random Coil)')
                    plt.grid(True)
                    plt.savefig(f"{atom_type}_secondary_cs.png")
                    plt.close()
                
                plt.xlabel('Residue ID')
                plt.ylabel(f'{atom_type} Chemical Shift (ppm)')
                plt.title(f'Chemical Shifts for {atom_type}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{atom_type}_predicted_cs.png")
                plt.close()

def main():
    parser = argparse.ArgumentParser(description='Predict chemical shifts from a PDB/CIF file')
    parser.add_argument('--pdb', type=str, default=None, required=False, help='Path to PDB/CIF file')
    parser.add_argument('--mace', type=str, default=None, help='Path to MACE model (defaults to MACE_DESCRIPTOR_MODEL)')
    parser.add_argument('--cs_models', type=str, nargs='+', default=None, help='Paths to CS prediction models (defaults to MACE_CS_MODELS)')
    parser.add_argument('--rmax', type=float, default=5.0, help='Maximum radius for environment extraction')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save intermediate results')
    parser.add_argument('--prefix', type=str, default='', help='Optional prefix for saved files')
    
    args = parser.parse_args()
    
    predictor = CSPredictor(
        mace_model_path=args.mace,
        cs_model_paths=args.cs_models,
        device=args.device
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
