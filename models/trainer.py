import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from torch_geometric.loader import DataLoader
from torch.optim.swa_utils import AveragedModel

import wandb
from data.descriptors_dataset import DescriptorsDataset
from utils.losses import LOSS_FUNCTIONS

# --- Utility Functions ---
def compute_confusion_matrix(all_targets, all_predictions, class_names, name):
    cm = confusion_matrix(all_targets, all_predictions)
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel="Predicted Labels", ylabel="True Labels", title=f"{name.capitalize()} Confusion Matrix")
    return fig


# --- Trainer Class ---

class Trainer():
    def __init__(self, 
                 model: nn.Module,
                 logger, 
                 train_dataset: DescriptorsDataset, 
                 test_dataset: DescriptorsDataset, 
                 lr: float=1e-2, 
                 batch_size:int=64, 
                 num_epochs:int=100,
                 optimizer=optim.Adam,
                 loss_func_name="CrossEntropyLoss",
                 checkpoint_name:str="test",
                 checkpoints_path:str = "models/checkpoints",
                 device="cpu",
                 ema_decay: float = 0.999,
                 lr_scheduler_type: str = "StepLR",  # Scheduler type
                 step_size: int = 10,  # StepLR: decrease every 10 epochs
                 gamma: float = 0.5,
                 total_iters:int=100,
                 clip_value=5.0,
                 test_every=1000,
                 ):
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=4)
        self.test_classes = test_dataset.get_targets()
        self.atoms = test_dataset.get_atoms()
        self.loss_func = LOSS_FUNCTIONS[loss_func_name]
        self.num_epochs = num_epochs
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.logger = logger
        self.clip_value = clip_value
        self.test_every = test_every
        
        self.checkpoints_path = checkpoints_path
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.checkpoint_name=checkpoint_name
        
        self.ema_model = AveragedModel(self.model, avg_fn=lambda ema_param, param, num_avg: ema_decay * ema_param + (1 - ema_decay) * param)
        self.lr_scheduler = self._initialize_scheduler(lr_scheduler_type, step_size, gamma, total_iters)

    def _initialize_scheduler(self, lr_scheduler_type, step_size, gamma, total_iters):
        """Initialize the learning rate scheduler."""
        warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=total_iters)

        if lr_scheduler_type == "StepLR":
            main_scheduler =  optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif lr_scheduler_type == "ReduceLROnPlateau":
            main_scheduler =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=gamma, patience=step_size)
            return main_scheduler
        elif lr_scheduler_type == "CosineAnnealingLR":
            main_scheduler =  optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        else:
            raise ValueError(f"Unknown lr_scheduler_type: {lr_scheduler_type}")
        return optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[total_iters])
        
    def _get_model_results(self, batch):
        descriptors, edges = batch.x.to(self.device), batch.edge_index.to(self.device)
        with torch.no_grad():
            outputs = self.ema_model(descriptors, edges)
        return outputs
        
    def train(self):
        for epoch in tqdm(range(self.num_epochs), "training model"):
            correct = 0
            total = 0
            losses = []
            for i, batch in enumerate(self.train_dataloader):
                descriptors, edges = batch.x.to(self.device), batch.edge_index.to(self.device)
                targets = batch.y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(descriptors, edges)
                targets = targets.squeeze(1).long() if str(self.loss_func) == 'CrossEntropyLoss()' else targets
                loss = self.loss_func(outputs, targets)
                losses.append(loss.item())
                self.logger.log({"training loss": loss.item()})
                
                # Calculate accuracy
                if len(self.test_classes) > 0:
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                                    
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                self.optimizer.step()
                self.ema_model.update_parameters(self.model)
            
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(sum(losses) / len(losses))  # Uses loss to decide when to reduce LR
                else:
                    self.lr_scheduler.step()  # Other schedulers step per epoch
            
            epoch_accuracy = 100 * correct / total if len(self.test_classes) > 0 else 0
            self.logger.log({
                "epoch": epoch,
                "epoch_accuracy": epoch_accuracy,
            })
            
            if epoch % self.test_every == 0:
                self.test(test=False)
                self.test(test=True)
            
        self.test(test=False)
        self.save_model(epoch+1)
        
    def save_model(self, epoch):
        folder_path = os.path.join(self.checkpoints_path, f"{self.checkpoint_name}_id_{self.logger.id}")
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{self.checkpoint_name}_epoch_{epoch}")
        self.model.cpu()
        self.ema_model.cpu()
        d = {
            "model_state_dict": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),  
        }
        if self.lr_scheduler is not None:
            d["lr_scheduler"] = self.lr_scheduler.state_dict()
        self.model.to(self.device)
        self.ema_model.to(self.device)
        torch.save(d, file_path)
        
        
    # ------------Classification
    
    def _test_classification(self, name, test=True):
        loader = self.test_dataloader if test else self.train_dataloader

        all_targets, all_predictions, correct_pred, total_pred = self._collect_predictions_and_targets(loader)

        logs = self._compute_classification_metrics(name, all_targets, all_predictions, correct_pred, total_pred)
        self._log_confusion_matrix(name, all_targets, all_predictions)

        return logs


    def _collect_predictions_and_targets(self, loader):
        idx_to_class = {self.test_classes[key]: key for key in self.test_classes}
        correct_pred = {classname: 0 for classname in self.test_classes}
        total_pred = {classname: 0 for classname in self.test_classes}
        all_targets = []
        all_predictions = []

        for batch in tqdm(loader, "Collecting predictions"):
            targets = batch.y.to(self.device)
            outputs = self._get_model_results(batch)
            _, predicted = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for label, prediction in zip(targets, predicted):
                cls_name = idx_to_class[label.item()]
                correct_pred[cls_name] += int(label == prediction)
                total_pred[cls_name] += 1

        return all_targets, all_predictions, correct_pred, total_pred


    def _compute_classification_metrics(self, name, all_targets, all_predictions, correct_pred, total_pred):
        idx_to_class = {self.test_classes[key]: key for key in self.test_classes}
        y_true = np.stack(all_targets)[:, 0]
        y_pred = np.stack(all_predictions)

        logs = {
            f"{name} accuracy {cls}": 100 * correct / total_pred[cls]
            for cls, correct in correct_pred.items()
        }
        logs[f"{name} total accuracy"] = 100 * sum(correct_pred.values()) / sum(total_pred.values())

        for metric_name in ["precision_score", "recall_score", "f1_score"]:
            metric_fn = getattr(metrics, metric_name)
            per_class = metric_fn(y_true, y_pred, average=None)
            weighted = metric_fn(y_true, y_pred, average="weighted")
            macro = metric_fn(y_true, y_pred, average="macro")
            prefix = metric_name.replace("_score", "")
            logs.update({
                f"{name} weighted {prefix}": weighted,
                f"{name} unweighted {prefix}": macro,
                **{f"{name} {prefix} {idx_to_class[idx]}": score for idx, score in enumerate(per_class)}
            })

        return logs


    def _log_confusion_matrix(self, name, all_targets, all_predictions):
        fig = compute_confusion_matrix(all_targets, all_predictions, list(self.test_classes.keys()), name)
        self.logger.log({f"{name} Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)
        
    # ------------Regression    
        
    def _test_regression(self, name, test=True):
        loader = self.test_dataloader if test else self.train_dataloader

        mse_dicts = {
            "mse": {}, "mae": {}, "dummy_mse": {}, "dummy_mae": {}
        }

        for batch in tqdm(loader, f"testing model ({name})"):
            batch_atoms, targets, outputs = self._process_batch(batch)

            mse_losses = (targets - outputs).pow(2)
            mae_losses = (targets - outputs).abs()
            dummy_mse = targets.pow(2)
            dummy_mae = targets.abs()

            for atom, mse, mae, d_mse, d_mae in zip(batch_atoms, mse_losses, mae_losses, dummy_mse, dummy_mae):
                for k, val in zip(mse_dicts.keys(), [mse, mae, d_mse, d_mae]):
                    mse_dicts[k].setdefault(atom, []).append(val)

        return self._compile_regression_logs(name, mse_dicts)


    def _process_batch(self, batch):
        targets = batch.y.to(self.device)
        outputs = self._get_model_results(batch)
        batch_atoms = np.concatenate([np.atleast_1d(atom) for atom in batch.atom])
        return batch_atoms, targets, outputs


    def _compile_regression_logs(self, name, loss_dicts):
        logs = {}

        def aggregate_losses(loss_dict, label):
            flat = list(chain.from_iterable(loss_dict.values()))
            agg = torch.cat(flat).mean()
            return agg.sqrt().item() if "rmse" in label else agg.item()

        def per_atom_losses(loss_dict, label):
            return {
                f"{label} {name} {atom} loss":
                    torch.cat(tensors).mean().sqrt().item() if "rmse" in label else torch.cat(tensors).mean().item()
                for atom, tensors in loss_dict.items()
            }

        logs.update(per_atom_losses(loss_dicts["mse"], "rmse"))
        logs.update(per_atom_losses(loss_dicts["mae"], "mae"))
        logs.update(per_atom_losses(loss_dicts["dummy_mse"], "dummy rmse"))
        logs.update(per_atom_losses(loss_dicts["dummy_mae"], "dummy mae"))

        logs[f"rmse {name} total loss"] = aggregate_losses(loss_dicts["mse"], "rmse")
        logs[f"mae {name} total loss"] = aggregate_losses(loss_dicts["mae"], "mae")
        logs[f"dummy rmse {name} total loss"] = aggregate_losses(loss_dicts["dummy_mse"], "dummy rmse")
        logs[f"dummy mae {name} total loss"] = aggregate_losses(loss_dicts["dummy_mae"], "dummy mae")

        return logs
        
        
    def test(self, test=True):
        name = "test" if test else "train"
        self.ema_model.eval()
        if len(self.test_classes) > 0:
            test_logs = self._test_classification(name, test=test)
        else:
            test_logs = self._test_regression(name, test=test)
        self.ema_model.train()
        self.logger.log(test_logs)
        
    





        
        
