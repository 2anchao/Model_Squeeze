#Author: Chao
cfgs = {
    "small_peleenet":{
        'num_init_features':32,
        'growthRate':32,
        'nDenseBlocks':[3,3],
        'bottleneck_width':[1,2],
        },
    "train":{
        'epochs':1,
        'train_batch_size':64,
        'num_classes':10,
    },
    "test_batch_size":256,
    'quant_bit': 4,
}