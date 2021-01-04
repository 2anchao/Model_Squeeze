#Author: Chao
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.peleenet_small import Peleenet32_Small
from utils.weight_pruning import MaskedConv2d
from configs import cfgs

def drow_weight(model):
    num_plot = 0
    modules = [module for module in model.modules() if isinstance(module, MaskedConv2d)]
    plt.figure(figsize=(12,6),dpi=100)
    for i, layer in enumerate(modules[:4]):
        if isinstance(layer, MaskedConv2d):
            plt.subplot(221+num_plot)
            w = layer.weight.data
            print(w)
            flat_w = w.cpu().numpy().flatten()
            plt.hist(flat_w, bins=50)
            num_plot += 1
    plt.show()

if __name__ == "__main__":
    num_classes = 10 
    ori_model = Peleenet32_Small(num_classes=num_classes)
    ori_checkpoint = torch.load("./save_models/mnist_origin_model.pth")
    ori_model.load_state_dict(ori_checkpoint["net"])
    quant_model = Peleenet32_Small(num_classes=num_classes)
    quant_checkpoint = torch.load("./save_models/mnist_pruning_model.pth")
    quant_model.load_state_dict(quant_checkpoint["net"])
    drow_weight(ori_model)
    drow_weight(quant_model)


