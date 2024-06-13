# add mask to file name
# n=1; for f in *.png; do mv -- "$f" "${f%.png}_$((n++))_mask.png"; done
# CUDA_VISIBLE_DEVICES=0 python train.py --epoch 10 --batch-size 4 --learning-rate 0.0001 --type bb-post_nas_quant_sqt --data carvana --name_append proxy_5k_epoch_10 &> logs/sqt/post_nas/bb-post_nas_quant_sqt_proxy_5k_epoch_10_train_batch_4_epoch_10_lr_0.0001_carvana
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.onnx
import onnx
from onnx import shape_inference

import matmul_pytorch


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = matmul_pytorch.matmul_workload() 
    
    model.to(device=device)
    #dummy_input=torch.randn(1,256, 112, 112).to(device)  
    dummy_input=torch.randn(1024, 1024).to(device)  

    #dummy_input=torch.randn(1,512, 112, 112).to(device)  

    #dummy_input=torch.randn(1,3, 112, 112).to(device)

    # x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

    # Export the model
    torch.onnx.export(model.eval(),               # model being run
                    dummy_input,                         # model input (or a tuple for multiple inputs)
                    "matmul.onnx", #"UNet_bottleneck_full_reduction_4_inchannel_3_without_bias.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    )
    model = onnx.load('matmul.onnx') #('UNet_bottleneck_full_reduction_4_inchannel_3_without_bias.onnx')
    onnx.save_model(model, 
                    'matmul.onnx', #'UNet_bottleneck_full_reduction_4_inchannel_3_without_bias_save.onnx', 
                    save_as_external_data=True, 
                    all_tensors_to_one_file=True,
                    location='external_data_filename', 
                    size_threshold=1024, 
                    convert_attribute=False)
    
    model = onnx.load('matmul.onnx') #("UNet_bottleneck_full_reduction_4_inchannel_3_without_bias_save.onnx")
    inferred_model = shape_inference.infer_shapes(model)
    onnx.save(inferred_model, "matmul.onnx") #"UNet_bottleneck_full_reduction_4_inchannel_3_without_bias_save_infer.onnx")
