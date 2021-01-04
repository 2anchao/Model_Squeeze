#Author: Chao
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)

def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """
    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i
    return min_v, min_ix

def get_model_masks(model):
    "Get mask use in model for pruning"
    values = []
    masks = []
    for name, module in model.named_modules():
        if isinstance(module, MaskedConv2d):
            weight = module.weight.data
            assert len(weight.size()) == 4 # (filter_number, channel, h, w)
            np_w = weight.cpu().numpy()
            masks.append(np.ones(np_w.shape).astype('float32'))
            # calculate scale L2 norm
            ## 1: square sum
            value_this_layer = np.square(np_w).sum(axis=1).sum(axis=1).sum(axis=1)/ \
            (np_w.shape[1]*np_w.shape[2]*np_w.shape[3])
            ## 2: norm
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_index = arg_nonzero_min(list(value_this_layer))
            # min_index = np.argmin(value_this_layer)
            # min_value = value_this_layer[min_index]
            values.append([min_value, min_index])
            
    assert len(masks) == len(values)
    values = np.array(values)
    # find L2 value min layer
    to_prune_layer_ind = np.argmin(values[:, 0])
    # find min correspond layer min L2 value correspond filter
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    # set correspond mask equal zero
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    return masks

def prune_rate(model, verbose=True):
    """
    print prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning conv layers
        if len(parameter.data.size()) == 4:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc

def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    '''
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = get_model_masks(model)
        model.use_pruning(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        print("current_pruning_perc:", current_pruning_perc)
    return masks

class MaskedConv2d(nn.Conv2d):
    '''
    convlution with pruning
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(MaskedConv2d, self).__init__(in_channels=in_channels, 
                                           out_channels=out_channels, kernel_size=kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation, groups=groups, 
                                           bias=bias)
        self.mask = None
        self.pruning_flag = False

    def pruning_with_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.pruning_flag = True
    
        
if __name__ == "__main__":
    test_weight = torch.randn(3,4)
    cluster_centers, labels = cluster_weight_cpu(weight=test_weight, cluster_K=3)
    reconstruct_weight = reconstruct_weight_from_cluster_result(cluster_centers, labels)
    pdb.set_trace()