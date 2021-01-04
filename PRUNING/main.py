#Author: Chao
import os
import torch
import math
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
from copy import deepcopy
from models.peleenet_small import Peleenet32_Small
from configs import cfgs
from utils.weight_pruning import filter_prune
import pdb

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total = 0
    for batch_idx, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total += len(img)
        print("Epoch:%d "%epoch,"Progress:%d/%d"%(total,len(train_loader.dataset))," Train Loss:%s"%str(round(loss.item(),4)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(test_loader):
            img, target = img.to(device), target.to(device)
            output = model(img)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            predict = output.argmax(dim=1, keepdim=True)
            correct += predict.eq(target.view_as(predict)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = correct/len(test_loader.dataset)
    print("Test Loss:%s"%str(round(test_loss, 4)), "Correct:%f"%acc)
    return acc

def main():
    epochs = cfgs["train"]["epochs"]
    train_batch_size = cfgs["train"]["train_batch_size"]
    test_batch_size = cfgs["test_batch_size"]
    num_classes=cfgs["train"]["num_classes"]
    quant_bit = cfgs["quant_bit"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(datasets.MNIST("./data/MNIST", 
                                                              train=True, 
                                                              download=True, 
                                                              transform=transforms.Compose(
                                                                        [transforms.ToTensor(),
                                                                        transforms.Normalize((0.1307,),(0.3081,))
                                                                                            ])),
                                                batch_size = train_batch_size,
                                                shuffle = True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST("./data/MNIST", 
                                                              train=False, 
                                                              download=True, 
                                                              transform=transforms.Compose(
                                                                        [transforms.ToTensor(),
                                                                        transforms.Normalize((0.1307,),(0.3081,))
                                                                                            ])),
                                                batch_size = test_batch_size,
                                                shuffle = False)
    model = Peleenet32_Small(num_classes=num_classes)
    optimizer = torch.optim.Adadelta(model.parameters())
    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)

    print("Origin model acc:%f"%acc)
    print("*.*"*10)

    print('\npruning 50%')
    pruning_model = deepcopy(model)
    masks= filter_prune(pruning_model, 15)
    pruning_model.use_pruning(masks)
    acc = test(pruning_model, device, test_loader)

    return model, pruning_model

if __name__ == "__main__":
    os.makedirs("save_models", exist_ok=True)
    model, pruning_model = main()
    state = {'net':model.state_dict()}
    pruning_state = {'net':pruning_model.state_dict()}
    torch.save(state,"./save_models/mnist_origin_model.pth")
    torch.save(pruning_state,"./save_models/mnist_pruning_model.pth")