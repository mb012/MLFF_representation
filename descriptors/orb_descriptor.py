
import pickle
from pathlib import Path
from .base_descriptor import BaseDescriptor
import os
import numpy as np
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs
from orb_models.forcefield.atomic_system import SystemConfig
from tqdm import tqdm



class OrbDescriptor(BaseDescriptor):
    def __init__(self, *args, **kwargs):
        super().__init__(desc_name="orb", *args, **kwargs)
        
    def get_model(self, model:str=""):
        return pretrained.orb_v2(device=self.device)
    
    def _calc_descriptors(self, model, atoms_obj_env, **kwargs):
        graph_rmax = kwargs["graph_rmax"]
        graph_nmax_neighbors = kwargs["graph_nmax_neighbors"]
        graph = atomic_system.ase_atoms_to_atom_graphs(atoms_obj_env, device=self.device, 
                                                    system_config=SystemConfig(radius=graph_rmax, max_num_neighbors=graph_nmax_neighbors))
        return model.model(graph).node_features['feat'].cpu().numpy()
    
    def _calc_batched_descriptors(self, model, batch):
        graph = batch_graphs(batch)
        return model.model(graph).node_features['feat'].cpu().numpy()
    
    def _reset_batch(self, atom_lst):
        return {f"{atom}":[] for atom in atom_lst}, [] ,[]
    
    def generate_batched_descriptors(self, data_folder:str, model_name:str, atom_lst:str="CA", rmax:float=5.0, num_workers:int=20,
                                     batch_size:int=10, filter_exisiting_files=True, graph_rmax:float=5.0, graph_nmax_neighbors:int=20):
        envs = self.get_envs(data_folder, rmax=rmax, num_workers=num_workers)
        envs = self._filter_exisiting_files(envs, atom_lst) if filter_exisiting_files else envs
        model = self.get_model(model_name)
        
        for atom in atom_lst:
            os.makedirs(Path(self.desc_path) / atom, exist_ok=True)        
        
        for amino_acid_env in tqdm(envs, desc="Computing descriptors"):
            prot_atom_descriptors = {f"{atom}":[] for atom in atom_lst}
            indices, batch, env_sizes = self._reset_batch(atom_lst)
            for i, (res_id, env) in enumerate(amino_acid_env.items()):
                atoms_obj_env = self._compute_atoms(env)
                graph = atomic_system.ase_atoms_to_atom_graphs(atoms_obj_env, device=self.device, 
                                                    system_config=SystemConfig(radius=graph_rmax, max_num_neighbors=graph_nmax_neighbors))
                batch += [graph]
                env = env.reset_index(drop=True)
                for atom in atom_lst:
                    indices[atom] += [env.index[(env["res_id"] == res_id) & (env["atom"] == atom)][0]]
                env_sizes += [env.shape[0]]
                if len(batch) == batch_size or i == len(amino_acid_env.items())-1:
                    descriptors = self._calc_batched_descriptors(model, batch)
                    for atom in atom_lst:
                        indices[atom] = np.cumsum(env_sizes)-np.array(env_sizes)+np.array(indices[atom])
                    descriptor_dict = {f"{atom}":descriptors[indices[atom]] for i,atom in enumerate(atom_lst)}
                    bmrb_id, pdb_id = env["bmrb_id"].unique()[0], env["pdb_id"].unique()[0]
                    for atom in descriptor_dict.keys():
                        prot_atom_descriptors[atom].append(
                        {
                            "bmrb_id": bmrb_id,
                            "pdb_id": pdb_id,
                            "descriptor": descriptor_dict[atom],
                        })
                    indices, batch, env_sizes = self._reset_batch(atom_lst)

            for atom in atom_lst:
                with open(f"{self.desc_path}/{atom}/{bmrb_id}_{pdb_id}.pkl", "wb") as f:
                    pickle.dump(prot_atom_descriptors[atom], f)
                    f.close()

    


