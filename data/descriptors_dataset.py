
from typing import Any


from collections.abc import Mapping
import pickle
import random
import re
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data


# currently supports 2 modes classification or regression with cs
class DescriptorsDataset(Dataset):
    def __init__(self, data_path, atoms, target, test=False, pka_residue="E"):
        entries = pd.read_csv("./data/test_set.csv") if test else pd.read_csv("./data/train_set.csv")
        self.target = target
        self.atoms = atoms

        assert target in ["cs", "secondary_structure", "pka", "name"]

        # filter entries with nan target
        self.dataset = []
        self.target_values = set()
        self.target_count_dict = {}
        self.protein_data = {}

        # for each protein go over relevant atoms and adds them according to residue (key) to the dictionary
        for file_name in tqdm(entries["bmrb_id"], "filtering entries"):
            amino_acid_data = {}
            for folder in os.listdir(data_path):
                if folder not in atoms or f"{file_name}.pkl" not in os.listdir(f"{data_path}/{folder}"):
                    continue
                with open(f"{data_path}/{folder}/{file_name}.pkl", "rb") as f:
                    data = pickle.load(f)
                for i, data_i in enumerate(data):
                    if folder in atoms and (pd.isnull(data_i[target]) or (pd.isnull(data_i["rc_cs"]) and target == "cs")):
                        continue
                    if target == "pka" and data_i["name"] != pka_residue:
                        # filtering only glu
                        continue
                    if target == "secondary_structure":
                        data_i[target] = self._adjust_secondary_structure_targets(data_i[target])
                    amino_acid_data.setdefault(f"{file_name},{data_i['res_id']}", []).append(data_i)
            
            regression_targets = ["cs", "pka"]

            for d in amino_acid_data.values():
                data_value = self._combine_aa_dict_list(d)
                if self.target == "cs":
                    data_value["cs"] = np.atleast_1d(data_value["cs"])
                    data_value["rc_cs"] = np.atleast_1d(data_value["rc_cs"])
                self.dataset.append(data_value)
                if target not in regression_targets:
                    # classification not regression
                    self.target_values.add(data_value[target])
                    self.target_count_dict.setdefault(data_value[target], 0)
                    self.target_count_dict[data_value[target]] += 1
        
        if "locohd" in data_path:
            # locohd is not a ff model so the descriptors are of a different size and cannot be batched - make all of them the same size
            self._fix_descriptor_size()
                    
        if len(self.target_values) != 0:
            self.target_values = sorted(self.target_values) 
            self.target_values = {val: i for i, val in enumerate(self.target_values)}
            print(self.target_count_dict)

        self._set_protein_data()
        
    def _fix_descriptor_size(self):
        unique_keys = sorted(set(
            key
            for d in self.dataset
            for desc in (d["descriptor"] if isinstance(d["descriptor"], list) else [d["descriptor"]])
            for key in desc.keys()
        ))
        for d in self.dataset:
            desc = d["descriptor"] if isinstance(d["descriptor"], list) else [d["descriptor"]]
            for key in unique_keys:
                for sample in desc:
                    if key not in sample:
                        sample[key] = 0.0
            # make sure same order
            desc = [{k: sample[k] for k in unique_keys} for sample in desc]
            d["descriptor"] = np.array(list(list(sample.values()) for sample in desc))


    def _set_protein_data(self):
        for i,d in enumerate(self.dataset):
            self.protein_data.setdefault(f'{d["bmrb_id"]}_{d["pdb_id"]}', []).append(i)
            
    def _combine_aa_dict_list(self, dict_list):
        result = dict_list[0].copy()
        
        result['cs'] = [result['cs']]
        result['rc_cs'] = [result['rc_cs']]
        # result['pka'] = [result['pka']]
        result['atom'] = [result['atom']]
        result['coord'] = [np.array(result['coord'])]
        result['descriptor'] = list([result['descriptor']])

        for d in dict_list[1:]:
            result['cs'].append(d['cs'])
            result['rc_cs'].append(d['rc_cs'])
            # result['pka'].append(d['pka'])
            result['atom'].append(d['atom'])
            result['coord'].append(np.array(d['coord']))
            result['descriptor'].append(d['descriptor'])
            
        result['descriptor'] = result['descriptor'] if isinstance(result['descriptor'][0], Mapping) else np.array(result['descriptor'])
        result['cs'] = np.array(result['cs'])
        result['rc_cs'] = np.array(result['rc_cs'])
        result['atom'] = np.array(result['atom'])
        result['coord'] = result['coord']
        # result['pka'] = np.array(result['pka'])
        return result
        
    def get_atoms(self):
        return self.atoms
    
    def get_targets(self):
        return self.target_values
                        
    def get_num_targets(self):
        return len(self.target_values) if len(self.target_values) > 0 else 1
    
    def get_dim(self):  
        return self.dataset[0]["descriptor"].shape[0] if len(self.dataset[0]["descriptor"].shape)==1 else self.dataset[0]["descriptor"].shape[1]
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        descriptor, edge_index, y = self.get_inp_target(data)
        atom = data["atom"]
        return Data(x=descriptor, edge_index=edge_index, y=y, atom=atom)
    
    def get_indices_per_protein(self, pdb_id):
        return self.protein_data[pdb_id]
    
    def get_inp_target(self, data):
        descriptor = torch.tensor(data["descriptor"]).to(torch.float32)
        num_nodes = 1 if len(descriptor.shape) == 1 else descriptor.shape[0]
        
        # Fully connected graph: every node is connected to every other node
        row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
        edge_index = torch.vstack([row.flatten(), col.flatten()])
        
        if len(descriptor.shape) == 1:
            descriptor = descriptor[None]
        if len(self.target_values) == 0:
            # RC CS is for calibration
            if self.target == "cs":
                # assumes a single atom
                atom_cs = data["cs"][np.where(data['atom'] == self.atoms[0])[0][0]]
                atom_rc_cs = data["rc_cs"][np.where(data['atom'] == self.atoms[0])[0][0]]
                y = torch.tensor(atom_cs-atom_rc_cs, dtype=torch.float32)
            else:
                y = torch.tensor(data[self.target], dtype=torch.float32)
        else:
            y = torch.tensor(self.target_values[data[self.target]], dtype=torch.float32)
        y = y[None].repeat(descriptor.shape[0])
        if len(y.shape) == 1:
            y = y[...,None]
        return descriptor, edge_index, y
    
    def _adjust_secondary_structure_targets(self, value):
        if value != "E" and value != "H":
            return "O" # stands for other
        return value

    