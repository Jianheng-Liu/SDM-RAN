import cv2
import os
import torch
import torchvision
import copy
import argparse
import numpy as np
from utils.sdm import CountRegressor, Resnet50FPN
from methods import MAPS, Scales, Transform, labelled_boxes, get_features, process_image, detect_target
from methods import siamese_rects, calculate_IoU
from PIL import Image

parser = argparse.ArgumentParser(description="Detection Demo Code")
parser.add_argument("-m", "--model_path", type=str, default="./data/models/FamNet.pth",
                    help="path to trained model")
parser.add_argument("-g", "--gpu-id", type=int, default=0,
                    help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
parser.add_argument("-c", "--category", type=str, default="5",
                    help="/Path/to/target/category/file/")

args = parser.parse_args()
example_file = "detect/" + args.category + "/support/"
target_file = "detect/" + args.category + "/query/"
result_file = "detect/" + args.category + "/result/"

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
regressor = CountRegressor(6, pool='mean')

if use_gpu:
    resnet50_conv.cuda()
    regressor.cuda()
    regressor.load_state_dict(torch.load(args.model_path))
else:
    regressor.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

resnet50_conv.eval()
regressor.eval()

images = None
target_boxes = None
features = []
ref_image_list = []
rects_list = []
output = []

shot = 0
ap_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # ap50-ap95
total = 0

for root, dirs, files in os.walk(example_file):
    for f in files:
        if f.split('.')[-1] == 'jpg':

            shot += 1
            img = cv2.imread(root + f)

            # Coco annotations
            bounding_boxes = labelled_boxes(f.split('.')[0])
            rects = list()
            bounding_box = bounding_boxes[0]
            rects.append([int(bounding_box[1]),
                          int(bounding_box[0]),
                          int(bounding_box[1] + bounding_box[3]),
                          int(bounding_box[0] + bounding_box[2])])

            image = Image.open(root + f)
            image.load()
            ref_image_list.append(image)
            rects_list.append(rects)

            sample = {'image': image, 'lines_boxes': rects}
            sample = Transform(sample)
            image, boxes = sample['image'], sample['boxes']

            images = image.unsqueeze(0)
            target_boxes = boxes.unsqueeze(0)

            with torch.no_grad():
                features.append(get_features(resnet50_conv, images, target_boxes, MAPS, Scales))

for root, dirs, files in os.walk(target_file):
    for f in files:
        if f.split('.')[-1] == 'jpg':

            total += 1

            input_image = target_file + f
            image = Image.open(input_image)
            image.load()
            input_image = copy.deepcopy(image)
            image = process_image(image)
            image = image.unsqueeze(0)

            # Detect
            with torch.no_grad():
                for i in range(len(features)):
                    all_features = detect_target(resnet50_conv, image, features[i], MAPS, Scales)
                    output.append(regressor(all_features))

            ref_img = ref_image_list[0]
            rects = rects_list[0]

            # Coco Annotations
            bounding_boxes = labelled_boxes(f.split('.')[0])
            coco = list()
            for bounding_box in bounding_boxes:
                coco.append([int(bounding_box[0]),
                             int(bounding_box[1]),
                             int(bounding_box[0] + bounding_box[2]),
                             int(bounding_box[1] + bounding_box[3])])

            pr_list = siamese_rects(input_image, output, ref_img, rects, coco, target_file, f, result_file + f)
            output = []

            img = np.array(image)
            img = img.astype(np.uint8)

            ap_count_temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            iou_max = 0.0

            for pr_rect in pr_list:

                pr_x1 = pr_rect[0]
                pr_x2 = pr_rect[1]
                pr_y1 = pr_rect[2]
                pr_y2 = pr_rect[3]
                cv2.rectangle(img, (pr_x1, pr_y1), (pr_x2, pr_y2), (0, 0, 255), 2, 4)

                for gt_rect in coco:

                    coco_x1 = gt_rect[1]
                    coco_y1 = gt_rect[0]
                    coco_x2 = gt_rect[3]
                    coco_y2 = gt_rect[2]

                    cv2.rectangle(img, (coco_x1, coco_y1), (coco_x2, coco_y2), (0, 255, 255), 2, 4)

                    iou = calculate_IoU(pr_rect, gt_rect)
                    if iou_max < iou:
                        iou_max = iou

                    for index in range(10):
                        if iou > (0.5 + (0.05 * index)):
                            ap_count_temp[index] = 1

            print("IOU: ", iou_max)
            for index in range(10):
                ap_count[index] += ap_count_temp[index]

if total > 0:
    ap50 = float(ap_count[0] / total)
    ap75 = float(ap_count[5] / total)
    sum = 0
    for index in range(10):
        sum += ap_count[index]
    ap = float(sum / 10 / total)
    print("The Average Precision (AP) Result for Category " + args.category)
    print("AP: %.4f"%ap)
    print("AP50: %.4f"%ap50)
    print("AP75: %.4f"%ap75)

