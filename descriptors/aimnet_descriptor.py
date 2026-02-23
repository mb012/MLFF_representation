
import torch
from .base_descriptor import BaseDescriptor
from aimnet2calc import AIMNet2ASE
from ase.calculators.calculator import all_changes 


NODE_FEAT_KEY = "aim"


class WrapperAIMNet2ASE(AIMNet2ASE):
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        self.atoms = atoms.copy()
        self.do_reset()
        self.uptade_tensors()

        if self.atoms.cell is not None and self.atoms.pbc.any():
            #assert self.base_calc.cutoff_lr < float('inf'), 'Long-range cutoff must be finite for PBC'
            cell = self.atoms.cell.array
        else:
            cell = None

        results = self.base_calc({
            'coord': torch.tensor(self.atoms.positions, dtype=torch.float32, device=self.base_calc.device),
            'numbers': self._t_numbers,
            'cell': cell,
            'mol_idx': self._t_mol_idx,
            'charge': self._t_charge,
            'mult': self._t_mult,
        }, forces='forces' in properties, stress='stress' in properties)
        for k, v in results.items():
            results[k] = v.detach().cpu().numpy()

        self.results['aim'] = results['aim']
        self.results['energy'] = results['energy']
        self.results['charges'] = results['charges']
        if 'forces' in properties:
            self.results['forces'] = results['forces']
        if 'stress' in properties:
            self.results['stress'] = results['stress']

class AimNetDescriptor(BaseDescriptor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(desc_name="aimnet", *args, **kwargs)
        
    def get_model(self, model:str='aimnet2'):
        model = WrapperAIMNet2ASE(model)
        # hacky - to get the node features output
        model.base_calc.keys_out += [NODE_FEAT_KEY]
        model.base_calc.atom_feature_keys += [NODE_FEAT_KEY]
        return model
        
    
    def _calc_descriptors(self, model, atoms_obj_env):
        model.calculate(atoms_obj_env) 
        return model.results[NODE_FEAT_KEY]              



        
        