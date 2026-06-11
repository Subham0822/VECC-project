import torch
import torch.nn as nn

def train_confidence(model,
                     loader,
                     optimizer):

    criterion = nn.BCELoss()

    for x,y in loader:

        pred = model(x)

        loss = criterion(pred,y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return loss.item()