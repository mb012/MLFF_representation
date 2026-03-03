# Representing local protein environments with machine learning force fields

Predicting NMR chemical shifts and acid dissociation constant (pKa) for titrable groups from protein structure using learned atomic **descriptors** (MACE, ORB, AIMNet2, LoCO‑HD) and a lightweight neural network classifier/regressor.

> **Reference paper:** [_Representing local protein environments with machine learning force fields_](https://arxiv.org/abs/2505.23354)
>
> **Authors:** Meital Bojan, Sanketh Vedula, Advaith Maddipatla, Nadav Bojan Sellam, Federico Napoli, Paul Schanda, Alex M. Bronstein.
>

---

## 1) Environment

Use the minimal environment.

```bash
conda env create -f environment.yml
conda activate forcefields
```

---

## 2) Project layout

```
force_fields-production/
├── main.py                    # Entry point for descriptor generation & experiments
├── cs_predict.py              # End-to-end chemical shift prediction on a PDB/CIF
├── pka_predict.py             # End-to-end pKa prediction on a PDB/CIF for titrable residues
├── experiment.py              # Experiment wrapper (dataset, model, trainer, wandb)
├── configs/                   # YAML configs (descriptor + experiments)
│   ├── mace_config.yaml
│   ├── orb_config.yaml
│   ├── aimnet_config.yaml
│   └── locohd_config.yaml
├── data/                      # Dataset handling and input files
├── descriptors/               # Descriptor implementations
├── models/                    # Model architectures and training scripts
└── utils/                     # Helper scripts (losses, preprocessing, etc.)
```

---


## 3) Data & models

- **Download**: Weights can be found in following [Dropbox link](https://www.dropbox.com/scl/fi/oo97sg240xfjsmyp3t40o/force_fields_data.zip?rlkey=caxh7qjkxm33jzvk926pxykxi&st=j5ai5cfc&dl=0)
- **Place data under**: `./data/`  

---

## 4) Usage overview

### A) Generate descriptors
Set the "af-bmrb-h-v3" and "target_fixed" from the Dropbox link under the data folder.
Run the descriptor generation according to the chosen configuration file with mode="descriptor":
```bash
python main.py --config [config_file]
```
This step will compute descriptors for all structures under the configured `data`.

### B) Train and evaluate
Train models based on precomputed descriptors according to the chosen configuration file with mode="experiments":
```bash
python main.py --config [config_file]
```
The results and checkpoints will be saved in `models/checkpoints/`.

### C) Predict chemical shifts on a PDB/CIF
To predict NMR chemical shifts for a given structure, use:
```bash
python cs_predict.py \
  --pdb path/to/structure.pdb \
  --rmax 5.0 \
  --output outputs/predictions.csv \
  --device cuda:0 \
  --save_dir outputs/ \
  --prefix "" \
  --mace path/to/mace_model.pt \
  --cs_models path/to/model1.ckpt path/to/model2.ckpt
```
> **Note:** Update the paths to the mace model and the trained chemical shift prediction models (`--cs_models`) before running predictions.

### D) Predict pKa on a PDB/CIF
To predict pKa for a given structure, use:
```bash
python pka_predict.py \
  --pdb path/to/structure.pdb \
  --atoms N CA C H HA CB
  --rmax 5.0 \
  --output outputs/predictions.csv \
  --device cuda:0 \
  --save_dir outputs/ \
  --prefix "" \
  --mace path/to/mace_model.pt \
  --pka_model_paths LYS=path/to/model1.ckpt HIS=path/to/model2.ckpt
```
> **Note:** Update the paths and keys to the mace model (optionally) and the trained pKa prediction models (`--pka_model_paths`) before running predictions.

---

## 5) Citation

If you use this repository, please cite the following paper:

```bibtex
@misc{bojan2026representinglocalproteinenvironments,
      title={Representing local protein environments with machine learning force fields}, 
      author={Meital Bojan and Sanketh Vedula and Advaith Maddipatla and Nadav Bojan Sellam and Federico Napoli and Paul Schanda and Alex M. Bronstein},
      year={2026},
      eprint={2505.23354},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2505.23354}, 
}
```

---

## 6) Tips

- Make sure `dssp` is installed if using secondary-structure features and the DSSP_PATH is correct.
- Check and modify configs under `configs/` to switch descriptors or tune hyperparameters.

