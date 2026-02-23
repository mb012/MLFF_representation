
from descriptors.base_descriptor import BaseDescriptor
from mace.calculators import mace_off
import matplotlib.pyplot as plt


class MaceDescriptor(BaseDescriptor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(desc_name="mace", *args, **kwargs)

    def get_model(self, model:str="large"):
        return mace_off(model=model, device=self.device)
    
    def _calc_descriptors(self, model, atoms_obj_env, **kwargs):
        invariants_only = kwargs["invariants_only"]
        num_layers = kwargs["num_layers"]
        return model.get_descriptors(atoms_obj_env, invariants_only=invariants_only, num_layers=num_layers)
    
    def visualize_mace_force(self, model, atoms_obj_env):
        forces = model.get_forces(atoms_obj_env)
        coordinates = atoms_obj_env.positions
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        u, v, w = forces[:, 0], forces[:, 1], forces[:, 2]
        
        forces2 = forces[:len(forces)//2]
        coordinates2 = coordinates[:len(forces)//2]

        x2, y2, z2 = coordinates2[:, 0], coordinates2[:, 1], coordinates2[:, 2]
        u2, v2, w2 = forces2[:, 0], forces2[:, 1], forces2[:, 2]

        # Create 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.quiver(x, y, z, u, v, w, length=0.3, normalize=True, color='b', alpha=0.5)
        ax.quiver(x2, y2, z2, u2, v2, w2, length=0.3, normalize=True, color='r', label="Forces 2", alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"MACE Force Vector Field {forces.shape[0]} atoms")

        plt.savefig(f"descriptors/visualizations/mace_forces_atoms_bigger.png", dpi=300)
      
   
       