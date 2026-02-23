
from data.descriptors_dataset import DescriptorsDataset
from models import NN
from models import Trainer
import wandb

class Experiment():
    def __init__(self,
                    wandb_key,
                    wandb_params,
                    dataset_params,
                    model_class,
                    model_params,
                    trainer_params,
                    ):
        
        
        train_dataset = DescriptorsDataset(**vars(dataset_params))
        test_dataset = DescriptorsDataset(**vars(dataset_params), test=True)
        print(f"Train dataset size: {len(train_dataset)}\t Test dataset size: {len(test_dataset)}")

        num_classes = train_dataset.get_num_targets()
        in_dim = train_dataset.get_dim()

        model = getattr(NN, model_class)
        model = model(**vars(model_params), in_dim=in_dim, out_dim=num_classes)
        
        def count_trainable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Example usage:
        print("Trainable parameters:", count_trainable_parameters(model))
        
        wandb.login(key=wandb_key)
        self.wandb = wandb.init(
            **vars(wandb_params),
            config={
                "dataset_params": {**vars(dataset_params)}, 
                "model_params": {**vars(model_params)}, 
                "trainer_params": {**vars(trainer_params)},
            }
        )
        self.trainer = Trainer(model, logger=self.wandb, train_dataset=train_dataset, test_dataset=test_dataset, **vars(trainer_params))
        
        
    def run(self):
        self.trainer.train()
        self.trainer.test()
        wandb.finish()
        
   
        
        

