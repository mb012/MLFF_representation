import torch.nn as nn

mse_loss = nn.MSELoss() 
l1_loss = nn.L1Loss()


LOSS_FUNCTIONS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "MSELoss": mse_loss,
    "L1Loss": l1_loss,
}
