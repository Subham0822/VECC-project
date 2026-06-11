import torch
import torch.nn as nn

def train_epoch(model,
                loader,
                optimizer):

    mse = nn.MSELoss()

    model.train()

    for x,y in loader:

        pred = model(x)

        loss = mse(pred,y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return loss.item()