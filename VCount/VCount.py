import copy
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import MincountLoss, PerturbationLoss,siamese

from PIL import Image
import cv2
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import torch.optim as optim

parser = argparse.ArgumentParser(description="Test code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the dataset")
parser.add_argument("-ts", "--split", type=str, default='visulaztion', choices=["test","val","visulaztion"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./util/SDM.pth", help="path to trained util")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
parser.add_argument("-shot",  "--shot", type=int, default=3, help="shot number.")
args = parser.parse_args()
data_path = args.data_path

anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'split.json'
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

cnt = 0

SAE_ER = 0
SSE_ER = 0

SAE_count = 0  # sum of absolute errors
SSE_count = 0  # sum of square errors
SAE_our = 0
SSE_our = 0

SAE_our_ = 0
SSE_our_ = 0

print("Test on {} data".format(args.split))
print(args.shot,'shot')
im_ids = data_split[args.split]
our_count = []
everthing_count = []
gt_count = []
N = 0

for im_id in im_ids:
    N+=1
    anno = annotations[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    image = Image.open('{}/{}'.format(im_dir, im_id))
    image.load()
    img = image.copy()
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    rects = list()
    for MM,bbox in enumerate(bboxes):
        x1, y1 = bbox[0][0], bbox[0][1]
        x2, y2 = bbox[2][0], bbox[2][1]
        rects.append([y1, x1, y2, x2])
        if MM == args.shot-1 :
            break

    sample = {'image': image, 'lines_boxes': rects}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']

    if use_gpu:
        image = image.cuda()
        boxes = boxes.cuda()

    with torch.no_grad(): features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
    if not args.adapt:
        with torch.no_grad(): output = regressor(features)
    else:
        features.required_grad = True
        adapted_regressor = copy.deepcopy(regressor)
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)
        for step in range(0, args.gradient_steps):
            optimizer.zero_grad()
            output = adapted_regressor(features)
            lCount = args.weight_mincount * MincountLoss(output, boxes)
            lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
            Loss = lCount + lPerturbation

            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()
        features.required_grad = False
        output = adapted_regressor(features)
    gt_cnt = dots.shape[0]
    pred_cnt = int(output.sum().item())
    count_s, box_s, object_num = siamese(img, output, rects, im_id,pred_cnt,gt_cnt,args.split)#第二阶段
    cnt = cnt + 1

    if dots.shape[0] > 30:
        object_num = pred_cnt

    er = abs(gt_cnt - object_num)
    SAE_ER +=  er
    SSE_ER += er**2

    print('------------------------')
    print('img_id:',im_id)
    print('GT value:',gt_cnt)
    print('Baseline(FamNet):', pred_cnt)
    print('Ours:', object_num)

    err_count = abs(gt_cnt - pred_cnt)
    SAE_count += err_count
    SSE_count += err_count ** 2

    err_our = abs(gt_cnt - object_num)
    SAE_our  +=  err_our
    SSE_our  +=  err_our ** 2

if args.split !='visulaztion':
    print('Baseline(FamNet): on {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.split, SAE_count / cnt,
                                                                             (SSE_count / cnt) ** 0.5))
    print('Our method on {} data,MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.split, SAE_ER / cnt, (SSE_ER/ cnt) ** 0.5))
else:
    print('Done!The results in visualization_results!')
