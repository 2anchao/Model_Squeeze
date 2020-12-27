#Author: Chao
import torch
from torch import nn
from sklearn.cluster import KMeans
import pdb

def cluster_weight_cpu(weight, 
                       cluster_K, 
                       init_method="k-means++", 
                       max_iter=30):
    '''
    Args:
    wieght: Tensor 
    cluster_K: The number of cluster center
    init_method: use in KMeans init, default k-means++
    max_iter: use in KMeans to limit max iteration
    Return:
    cluster center values
    cluster labels
    '''
    ori_shape = weight.shape
    weight = weight.view(-1, 1)

    kmeans = KMeans(n_clusters=cluster_K, init=init_method, max_iter=max_iter)
    kmeans.fit(weight)

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    labels = labels.reshape(ori_shape)
    return torch.as_tensor(cluster_centers).view(-1, 1), torch.as_tensor(labels, dtype=torch.int8)

def reconstruct_weight_from_cluster_result(cluster_centers, labels):
    '''
    Args:
    cluster_centers: cluster_centers from KMeans cluster
    labels: labels from KMeans cluster
    Return:
    reconstruct_weight
    '''
    weight = torch.zeros_like(labels).float()
    for i, c in enumerate(cluster_centers):
        weight[labels==i] = c.item()
    return weight

class QuantLinear(nn.Linear):
    '''
    Use KMeans cluster to quant Linear layers' weight & bias
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(QuantLinear, self).__init__(in_features=in_features,
                                          out_features=out_features, 
                                          bias=bias)
        self.quant_weight = False
        self.quant_bias = False
        self.weight_labels = None
        self.weight_center = None
        self.bias_labels = None
        self.bias_cneter = None
        self.num_centers = None

    def kmeans_quant(self, quant_bias=False, quant_bit=2):
        self.num_centers = 2**quant_bit
        self.quant_weight = True
        weight = self.weight.data
        self.weight_centers, self.weight_labels = cluster_weight_cpu(weight, self.num_centers)
        w_q = reconstruct_weight_from_cluster_result(self.weight_centers, self.weight_labels)
        self.weight.data = w_q.float()
        if quant_bias:
            self.quant_bias = True
            bias = self.bias.data
            self.bias_centers, self.bias_labels = cluster_weight_cpu(bias, self.num_centers)
            b_q = reconstruct_weight_from_cluster_result(self.bias_centers, self.bias_labels)
            self.bias.data = b_q.float()
    
class QuantConv2d(nn.Conv2d):
    '''
    Use KMeans cluster to quant Conv2d layers' weight & bias
    '''
    def __init__(self, in_channels, 
                out_channels, kernel_size, stride=1, 
                padding=1, dilation=1, groups=1, bias=True):
        super(QuantConv2d, self).__init__(in_channels=in_channels, 
                                          out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                          padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.quant_weight = False
        self.quant_bias = False
        self.weight_labels = None
        self.weight_center = None
        self.bias_labels = None
        self.bias_cneter = None
        self.num_centers = None

    def kmeans_quant(self, quant_bias=False, quant_bit=2):
        self.num_centers = 2**quant_bit
        self.quant_weight = True
        weight = self.weight.data
        self.weight_centers, self.weight_labels = cluster_weight_cpu(weight, self.num_centers)
        w_q = reconstruct_weight_from_cluster_result(self.weight_centers, self.weight_labels)
        self.weight.data = w_q.float()
        if quant_bias:
            self.quant_bias = True
            bias = self.bias.data
            self.bias_centers, self.bias_labels = cluster_weight_cpu(bias, self.num_centers)
            b_q = reconstruct_weight_from_cluster_result(self.bias_centers, self.bias_labels)
            self.bias.data = b_q.float()        


if __name__ == "__main__":
    test_weight = torch.randn(3,4)
    cluster_centers, labels = cluster_weight_cpu(weight=test_weight, cluster_K=3)
    reconstruct_weight = reconstruct_weight_from_cluster_result(cluster_centers, labels)
    pdb.set_trace()