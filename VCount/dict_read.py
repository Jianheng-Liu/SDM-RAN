
#!/usr/bin/env python
# coding: utf-8

"""
Test code written by Viresh Ranjan

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""

import copy
from model import CountRegressor, Resnet50FPN
#from util import MAPS, Scales, Transform, extract_features
#from util import MincountLoss, PerturbationLoss,siamese

from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import torch.optim as optim
import torchvision

parser = argparse.ArgumentParser(description="Test code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["test","val","show"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./util/model1.pth", help="path to trained util")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
parser.add_argument("-shot",  "--shot", type=int, default=1, help="shot number.")
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'

if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
if use_gpu: resnet50_conv.cuda()
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean')
regressor.load_state_dict(torch.load(args.model_path))
if use_gpu: regressor.cuda()
regressor.eval()

with open(anno_file) as f:
    annotations = json.load(f)
with open(data_split_file) as f:
    data_split = json.load(f)

name = os.listdir('our_dot_s')
data_split['show_sample'] = name
del data_split['test_coco']
del data_split['val_coco']


with open('split.json','w') as f:
    json.dump(data_split, f)


