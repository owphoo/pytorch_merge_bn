import torch
import os
from collections import OrderedDict
import cv2
import numpy as np

def merge(params, name, layer, deconv_layer_names=['deconv']):
    # global variables
    global weights, bias
    global bn_param
    global merged

    is_deconv = False
    for deconv_name in deconv_layer_names:
        if deconv_name in name:
            is_deconv = True
            break 
    if layer == 'Convolution':
        # save weights and bias when meet conv layer
        if 'weight' in name:
            weights = params.data
            bias = torch.zeros(weights.size()[0], device=weights.device)
            if is_deconv:
                bias = torch.zeros(weights.size()[1], device=weights.device)
            else:
                bias = torch.zeros(weights.size()[0], device=weights.device)
            merged = False
        elif 'bias' in name:
            bias = params.data
        bn_param = {}

    elif layer == 'BatchNorm':
        # save bn params
        bn_param[name.split('.')[-1]] = params.data

        # running_var is the last bn param in pytorch
        if 'running_var' in name:
            # merge bn
            tmp = bn_param['weight'] / torch.sqrt(bn_param['running_var'] + 1e-5)
            if is_deconv:
                weights = (tmp.view(tmp.size()[0], 1, 1, 1) * weights.permute(1,0,2,3)).permute(1,0,2,3)
            else:
                weights = tmp.view(tmp.size()[0], 1, 1, 1) * weights
            bias = tmp*(bias - bn_param['running_mean']) + bn_param['bias']

            return weights, bias

    return None, None


import sys
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python pytorch_merge_bn.py YOU_MODEL')
        sys.exit(-1)
    model_path = sys.argv[1]
    print('input model: ', model_path)
    checkpoint = torch.load(model_path)

    trained_weights = checkpoint['net_state_dict']

    '''
    ## conv_bn_relu module
    #           NAME           |           SIZE
    #   conv4.0.weight              torch.Size([128, 256, 3, 3])
    #   conv4.1.weight              torch.Size([256])
    #   conv4.1.bias                torch.Size([256])
    #   conv4.1.running_mean        torch.Size([256])
    #   conv4.1.running_var         torch.Size([256])


    ## deconv_bn_relu module
    #           NAME           |           SIZE
    #   deconv4.0.weight             torch.Size([256, 128, 4, 4])
    #   deconv4.1.weight             torch.Size([128])
    #   deconv4.1.bias               torch.Size([128])
    #   deconv4.1.running_mean       torch.Size([128])
    #   deconv4.1.running_var        torch.Size([128])
    '''

    # check it in your net modules
    deconv_layer_names = ['deconv4', 'deconv3', 'deconv2', 'deconv1']
    temp = []
    for deconv_name in deconv_layer_names:
        temp.append(deconv_name + '.0')
        temp.append(deconv_name + '.1')
    deconv_layer_names = temp

    # go through pytorch net
    new_weights = OrderedDict()
    inner_product_flag = False
    for name, params in trained_weights.items():
        print ('name: ', name, params.size())
        if len(params.size()) == 4:
            _, _ = merge(params, name, 'Convolution', deconv_layer_names=deconv_layer_names)
            prev_layer = name
            # print('prev1: ', prev_layer)
        elif len(params.size()) == 1 and not inner_product_flag:
            w, b = merge(params, name, 'BatchNorm', deconv_layer_names=deconv_layer_names)
            # print('prev2: ', prev_layer)
            if w is not None:
                new_weights[prev_layer] = w
                new_weights[prev_layer.replace('weight', 'bias')] = b
            # mergebn
            merged = True
        else:
            # inner product layer (TODO, inner product layer may have bn module)
            if name.find('num_batches_tracked') == -1:
                new_weights[name] = params
                inner_product_flag = True
            else:
                pass

    # for the last conv/deconv if it has no bn module 
    if merged is False:
        new_weights[prev_layer] = weights
        new_weights[prev_layer.replace('weight', 'bias')] = bias

    checkpoint['net_state_dict'] = new_weights

    # save new weights
    model_name = model_path[model_path.rfind('/')+1:]
    model_path = model_path[:model_path.rfind('/')]
    torch.save(checkpoint, model_path + '/merge_bn_' + model_name)
