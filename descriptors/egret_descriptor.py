from descriptors.base_descriptor import BaseDescriptor
from mace.calculators import mace_off
import torch
import numpy as np
import e3nn.o3 as o3

# from https://github.com/rowansci/egret-public/blob/master/compiled_models/EGRET_1.model
EGRET_MODELS = ["./compiled_models/EGRET_1.model"]

class EgretDescriptor(BaseDescriptor):

    def __init__(self, *args, **kwargs):
        super().__init__(desc_name="egret", *args, **kwargs)

    def get_model(self, model:str=EGRET_MODELS[0]):
        return mace_off(model=model, device=self.device)

    def _calc_descriptors(self, model, atoms_obj_env, **kwargs):
        full_descriptor = model.models[0](model._atoms_to_batch(atoms_obj_env).to_dict())["node_feats"]
        invariant_descriptor = torch.cat([full_descriptor[:, :192], full_descriptor[:, -192:]], dim=1)
        return invariant_descriptor.clone().detach().cpu().numpy()

# if __name__ == "__main__":
#     descriptor = EgretDescriptor()
#     descriptor.generate_descriptors(
#         data_folder="./af_data/",
#         model_name=EGRET_MODELS[0],
#         num_workers=1,
#         filter_exisiting_files=False,
#         rmax=5.0
#     )
