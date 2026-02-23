
from collections import defaultdict
import os
from pathlib import Path
import pickle
from sklearn.neighbors import KDTree

import numpy as np
from tqdm import tqdm
from descriptors.base_descriptor import BaseDescriptor
import matplotlib.pyplot as plt 
from loco_hd import *
from Bio.PDB.PDBParser import PDBParser

def uniform_cdf(x, min_val, max_val):
    if x < min_val:
        return 0.0
    elif x > max_val:
        return 1.0
    else:
        return (x - min_val) / (max_val - min_val)

class WeightFunction:
    def __init__(self, function_name, parameters):
        self.function_name = function_name
        self.parameters = parameters

    def __call__(self, distance):
        # Call the appropriate function based on the name
        if self.function_name == "uniform":
            # parameters: [min_dist, max_dist]
            min_dist, max_dist = self.parameters
            return 1.0 if min_dist <= distance <= max_dist else 0.0

        else:
            raise NotImplementedError("Only uniform weight function is implemented")
        
    def integral_range(self, a, b):
        if self.function_name == "uniform":
            k = 1 / (self.parameters[1] - self.parameters[0])
            integral = k * max(0, min(b, self.parameters[1]) - max(a, self.parameters[0]))
            return integral
        else:
            raise ValueError("Unknown weight function type")

def get_weighted_distributions(prim_a, anchor_indices, threshold_distance, weight_func, tag_pairing_rule=None):
    coords = np.array([atom.coordinates for atom in prim_a])
    kdtree = KDTree(coords, metric="euclidean")
    
    distributions = []
    for anchor_idx in anchor_indices:
        anchor_atom = prim_a[anchor_idx]
        neighbors_idx = kdtree.query_radius(np.array(anchor_atom.coordinates).reshape(1, -1), r=threshold_distance)[0]
        neighbors_idx = [i for i in neighbors_idx if prim_a[i].tag!=anchor_atom.tag]
        
        env_types = [prim_a[i].primitive_type for i in neighbors_idx]
        env_dists = [np.linalg.norm(np.array(anchor_atom.coordinates) - np.array(prim_a[i].coordinates)) for i in neighbors_idx]
        combined = list(zip(env_dists, env_types))
        combined.sort()  # Sort based on distances
        env_dists, env_types = zip(*combined)
        env_dists, env_types = list(env_dists), list(env_types)
        
        # Compute weighted distribution
        weighted_delta_w = defaultdict(list)
        dist_per_r = defaultdict(list)
        curr_count = {k: 0.0  for k in set(env_types)}
        prev_dist = 0.0
        for d,t in zip(env_dists,env_types):
            curr_count[t] += 1.0
            normalized_curr_count = {k:v/sum(list(curr_count.values())) for k,v in curr_count.items()}
            dist_per_r[t].append(normalized_curr_count)
            delta_w = weight_func.integral_range(prev_dist, d)
            weighted_delta_w[t].append(delta_w)
            prev_dist = d
        sum_w = sum(sum(lst) for lst in weighted_delta_w.values())
        for t in weighted_delta_w.keys():
            weighted_delta_w[t] = sum([w * sum(list(dist.values())) for w, dist in zip(weighted_delta_w[t], dist_per_r[t])])
        weighted_delta_w = {k: v/sum_w for k,v in weighted_delta_w.items()}
        weighted_delta_w = dict(sorted(weighted_delta_w.items()))
        distributions.append(weighted_delta_w)
    return distributions

class LocoHDDescriptor(BaseDescriptor):
    
    def __init__(self, *args, **kwargs):
        print("running locohd descriptors")
        primitive_typing_path = kwargs["primitive_typing"]
        kwargs.pop("primitive_typing")
        max_threshold = kwargs["max_threshold"]
        kwargs.pop("max_threshold")
        min_threshold = kwargs["min_threshold"]
        kwargs.pop("min_threshold")
        super().__init__(desc_name="locohd_false_tag", *args, **kwargs)
        self.primitive_assigner = PrimitiveAssigner(Path(primitive_typing_path))
        self.primitive_typing_path = primitive_typing_path
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        assert max_threshold > min_threshold
        self.w_func = WeightFunction("uniform", [min_threshold, max_threshold])
        
    def _extract_amino_acid_env(self, df, rmax=5.0):
        file = df["af_file_name"][0]
        structure = PDBParser(QUIET=True).get_structure("s1", file)
        pra_templates = self.primitive_assigner.assign_primitive_structure(structure)
        env = {"pra_templates":pra_templates, "bmrb_id": df["bmrb_id"][0], "pdb_id":df["pdb_id"][0]}
        return env


    def get_model(self, model:str=""):
        return LoCoHD(self.primitive_assigner.all_primitive_types)

    
    def _calc_descriptors(self, model, pra_templates, **kwargs):
        anchor_pairs = [
            idx
            for idx, prat in enumerate(pra_templates)
            #if prat.primitive_type == "Cent"
        ]
        pra = kwargs["pra"]
        return get_weighted_distributions(pra,anchor_pairs,self.max_threshold, self.w_func)
    
    
    def generate_descriptors(self, data_folder:str, model_name:str, rmax:float=5.0, 
                             num_workers:int=20, filter_exisiting_files=True, **kwargs):
        envs = self.get_envs(data_folder, rmax=rmax, num_workers=num_workers)
        envs = self._filter_exisiting_files(envs) if filter_exisiting_files else envs
        model = self.get_model(model_name)
        
        for amino_acid_env, features in tqdm(envs, desc="Computing descriptors"):
            prot_atom_descriptors = defaultdict(list)
            pra_templates = amino_acid_env["pra_templates"]
            pra = list(map(prat_to_pra, pra_templates))
            kwargs["pra"] = pra
            descriptors = self._calc_descriptors(model, pra_templates, **kwargs)
            for i, env in enumerate(pra_templates):
                res_id = env.atom_source.source_residue[-1][1]
                bmrb_id, pdb_id = amino_acid_env["bmrb_id"], amino_acid_env["pdb_id"]
                atom = "".join(env.atom_source.source_atom)
                if env.primitive_type != 'Cent':
                    targets = self._get_targets(f"./data/targets/{bmrb_id}_{pdb_id}.csv", res_id, features, env.atom_source.source_atom)
                else:
                    targets = self._get_targets(f"./data/targets/{bmrb_id}_{pdb_id}.csv", res_id, features, None)
                prot_atom_descriptors[atom].append(
                {
                    "bmrb_id": bmrb_id,
                    "pdb_id": pdb_id,
                    "descriptor": descriptors[i],
                    "name": targets["name"],
                    "secondary_structure": targets["secondary_structure"],
                    "cs": targets["cs"][atom] if len(targets["cs"]) else np.nan,
                    "rc_cs": targets["rc_cs"][atom] if len(targets["rc_cs"]) else np.nan,
                    "res_id": res_id,
                    "atom": atom,
                    "coord": env.coordinates
                })
            
            primitive_name = self.primitive_typing_path.split("/")[-1].split(".")[0]
            for atom in prot_atom_descriptors.keys():
                os.makedirs(Path(self.desc_path) / primitive_name / atom, exist_ok=True)        
                filename = f"{bmrb_id}_{pdb_id}"
                with open(f"{self.desc_path}/{primitive_name}/{atom}/{filename}.pkl", "wb") as f:
                    pickle.dump(prot_atom_descriptors[atom], f)
                    f.close()
       
def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:
        resi_id = prat.atom_source.source_residue
        resname = prat.atom_source.source_residue_name
        source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"

        return PrimitiveAtom(
            prat.primitive_type, 
            source,  # this is the tag field!
            prat.coordinates
        )


                     

