import copy
import numpy as np
import torch.nn.functional as F
import math
from torchvision import transforms
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from utils.ran import self_model
import matplotlib.patches as patches


MAPS = ['map3', 'map4']
Scales = [0.9, 1.1]
FEATURE_SHAPE = 24
SIAMESE_MAGNIFICATION = 1.1
EXTRACT_MAFNIFICATION = 2.0
MERGE_DISTANCE = 128
MERGE_RATIO = 1.0
NOISE_SIZE = 24
MIN_DENSITY_THRESHOLD = 0.35
QUERY_PADDING_RATIO = 0.15
CONFIDENCE_THRESHOLD = 0.9
IGNORE_RATIO = 0.50
PRINT = True
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def get_features(feature_model, image, target_boxes, feat_map_keys=['map3', 'map4'],
                 exemplar_scales=[0.9, 1.1]):
    N, M = image.shape[0], target_boxes.shape[2]

    # Getting features for the example image N * C * H * W
    Image_features = feature_model(image)

    # Getting features for the examples (N*M) * C * h * w
    max_h = 0
    max_w = 0

    for ix in range(0, N):
        boxes = target_boxes[ix][0]
        image_features = Image_features['map3'][ix].unsqueeze(0)

        boxes_scaled = boxes / 8.0
        boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
        boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
        boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1  # make the end indices exclusive
        feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
        # make sure exemplars don't go out of bound
        boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
        boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
        boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)
        box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
        box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]
        h = math.ceil(max(box_hs))
        w = math.ceil(max(box_ws))

        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

    compress = FEATURE_SHAPE / max(max_h, max_w)

    features = []

    for keys in feat_map_keys:
        if keys == 'map1' or keys == 'map2':
            Scaling = 4.0
        elif keys == 'map3':
            Scaling = 8.0
        elif keys == 'map4':
            Scaling = 16.0
        else:
            Scaling = 32.0

        max_h = 0
        max_w = 0

        examples_features = None

        for ix in range(0, N):
            boxes = target_boxes[ix][0]
            image_features = Image_features[keys][ix].unsqueeze(0)

            boxes_scaled = boxes / Scaling
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1  # make the end indices exclusive
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)
            box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
            box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]
            h = math.ceil(max(box_hs))
            w = math.ceil(max(box_ws))

            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w

        for ix in range(0, N):
            boxes = target_boxes[ix][0]
            image_features = Image_features[keys][ix].unsqueeze(0)

            boxes_scaled = boxes / Scaling
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1  # make the end indices exclusive
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)

            for j in range(0, M):
                y1, x1 = int(boxes_scaled[j, 1]), int(boxes_scaled[j, 2])
                y2, x2 = int(boxes_scaled[j, 3]), int(boxes_scaled[j, 4])
                # print("Getting Features y1, y2, x1, x2, max_h, max_w: ", y1, y2, x1, x2, max_h, max_w)
                # examples_features
                if examples_features == None:
                    examples_features = image_features[:, :, y1:y2, x1:x2]
                    if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                        # examples_features = pad_to_size(examples_features, max_h, max_w)
                        examples_features = F.interpolate(examples_features, size=(max_h, max_w), mode='bilinear')
                    # while examples_features.shape[2] > 16 or examples_features.shape[3] > 16:
                    examples_features = F.interpolate(examples_features, scale_factor=compress, mode='nearest')
                else:
                    feat = image_features[:, :, y1:y2, x1:x2]
                    if feat.shape[2] != max_h or feat.shape[3] != max_w:
                        feat = F.interpolate(feat, size=(max_h, max_w), mode='bilinear')
                        # feat = pad_to_size(feat, max_h, max_w)
                    feat = F.interpolate(feat, scale_factor=compress, mode='nearest')
                    examples_features = torch.cat((examples_features, feat), dim=0)

        features.append(examples_features)

    return features

def detect_target(feature_model, image, input_features, feat_map_keys=['map3', 'map4'], exemplar_scales=[0.9, 1.1]):
    # exemplar_scales=[0.9, 1.1]

    N = image.shape[0]

    # Getting features for the image N * C * H * W
    Image_features = feature_model(image)

    # Getting features for the examples (N*M) * C * h * w
    for ix in range(0, N):
        cnter = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)
            # print("Image Features Shape: ", image_features.shape)
            # print(image_features)

            examples_features = None

            if keys == 'map3':
                examples_features = input_features[0]

            elif keys == 'map4':
                examples_features = input_features[1]

            # Convolving example features over image features
            h, w = examples_features.shape[2], examples_features.shape[3]
            # print("Example Feature: ", examples_features)

            features = F.conv2d(
                F.pad(image_features, ((int(w / 2)), int((w - 1) / 2), int(h / 2), int((h - 1) / 2))),
                examples_features
            )

            combined = features.permute([1, 0, 2, 3])
            # computing features for scales 0.9 and 1.1
            for scale in exemplar_scales:
                h1 = math.ceil(h * scale)
                w1 = math.ceil(w * scale)
                if h1 < 1:  # use original size if scaled size is too small
                    h1 = h
                if w1 < 1:
                    w1 = w
                examples_features_scaled = F.interpolate(examples_features, size=(h1, w1), mode='bilinear')
                # print("Examples Features Scaled Shape: ", examples_features_scaled.shape)

                padded = F.pad(image_features, ((int(w1 / 2)), int((w1 - 1) / 2), int(h1 / 2), int((h1 - 1) / 2)))
                # print("Padded Scaled Shape: ", padded.shape)

                features_scaled = F.conv2d(
                    F.pad(image_features, ((int(w1 / 2)), int((w1 - 1) / 2), int(h1 / 2), int((h1 - 1) / 2))),
                    examples_features_scaled)
                features_scaled = features_scaled.permute([1, 0, 2, 3])
                combined = torch.cat((combined, features_scaled), dim=1)
            if cnter == 0:
                Combined = 1.0 * combined
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2], Combined.shape[3]), mode='bilinear')
                Combined = torch.cat((Combined, combined), dim=1)
            cnter += 1

        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat, Combined.unsqueeze(0)), dim=0)
    return All_feat


def labelled_boxes(img_name, img_info_val='./data/annotations/coco_val.npy',
                   img_info_train='./data/annotations/coco_train.npy'):
    img_in = np.load(img_info_val, allow_pickle=True)

    for info in img_in:
        if info[0].split('.')[0] == img_name:
            return info[2]

    img_in = np.load(img_info_train, allow_pickle=True)

    for info in img_in:
        if info[0].split('.')[0] == img_name:
            return info[2]

    print("Not find target bounding boxes.")


def load_features(load_path):
    with torch.no_grad():
        features = np.load(load_path)

    return features


def process_image(img, max_hw=1504):
    W, H = img.size
    if W > max_hw or H > max_hw:
        scale_factor = float(max_hw) / max(H, W)
        new_H = 8 * int(H * scale_factor / 8)
        new_W = 8 * int(W * scale_factor / 8)
        resized_image = transforms.Resize((new_H, new_W))(img)
    else:
        # scale_factor = 1
        resized_image = img

    resized_image = Normalize(resized_image)

    return resized_image


class resizeImage(object):

    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, lines_boxes = sample['image'], sample['lines_boxes']

        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw) / max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k * scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image': resized_image, 'boxes': boxes}
        return sample


class resizeImageWithGT(object):

    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, lines_boxes, density = sample['image'], sample['lines_boxes'], sample['gt_density']

        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw) / max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)

        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k * scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image': resized_image, 'boxes': boxes, 'gt_density': resized_density}
        return sample


Normalize = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
Transform = transforms.Compose([resizeImage(MAX_HW)])
TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):

    new_val = int(round(val * scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val


def siamese(img, density, rects, im_id):
    density = torch.squeeze(density, axis=0)
    density = torch.squeeze(density, axis=0)
    density = density.cpu().numpy() * 50000
    density_ = density.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    density = cv2.morphologyEx(density_, cv2.MORPH_OPEN, kernel)
    # density =  cv2.equalizeHist(density)
    img = np.array(img)
    img = img.astype(np.uint8)
    img_o = img.copy()
    rett, bina_density = cv2.threshold(density, np.max(density) * 0.3, 255, cv2.THRESH_BINARY)  # 0.4
    # bina_density = cv2.adaptiveThreshold(density, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    # bina_density = cv2.adaptiveThreshold(density, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    # bina_density = cv2.adaptiveThreshold(density, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3,1)
    contours, hierarchy = cv2.findContours(bina_density, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        # cv2.drawContours(img, c, -1, (0, 0, 255), 2)
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])  # 求x坐标
        cy = int(M['m01'] / M['m00'])  # 求y坐标
        img = cv2.circle(img, (cx, cy), 2, (0, 0, 255), 4)  # 画出重心
        centers.append([cx, cy])
    w, h = 0, 0
    Gt_center_ratio = []
    for point in rects:
        y1, x1, y2, x2 = point
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        w += x2 - x1
        h += y2 - y1
        box_ratio = np.abs(np.abs(y2 - y1) - np.abs(x2 - x1))
        Gt_center_ratio.append(box_ratio)
    max_index = Gt_center_ratio.index(min(Gt_center_ratio))

    w, h = int(w / 3), (h / 3)
    Gt_center_point = [int((rects[max_index][3] + rects[max_index][1]) / 2),
                       int((rects[max_index][2] + rects[max_index][0]) / 2)]
    length = max(w, h) * 1.1  # 放大倍数
    Gt_x1 = int(Gt_center_point[0] - length / 2)
    Gt_x2 = int(Gt_center_point[0] + length / 2)
    Gt_y1 = int(Gt_center_point[1] - length / 2)
    Gt_y2 = int(Gt_center_point[1] + length / 2)
    # cv2.rectangle(img, (Gt_x1, Gt_y1), (Gt_x2, Gt_y2), (255, 255, 255), 2)
    Gt_point = [Gt_x1, Gt_x2, Gt_y1, Gt_y2]

    Pr_points = []

    for center in centers:
        x1 = int(center[0] - length / 2)
        x2 = int(center[0] + length / 2)
        y1 = int(center[1] - length / 2)
        y2 = int(center[1] + length / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        Pr_points.append([x1, x2, y1, y2])

    Gt_img = img_o[Gt_y1:Gt_y2, Gt_x1:Gt_x2, :]
    # Gt_img = cv2.cvtColor(Gt_img, cv2.COLOR_BGR2RGB)
    Gt_img = Image.fromarray(Gt_img)
    for Pr_point in Pr_points:
        imgc = img.copy()
        x11, x22, y11, y22 = Pr_point
        Pr_img = img_o[y11:y22, x11:x22, :]
        Pr_w, Pr_h = Pr_img.shape[0], Pr_img.shape[1]
        center_x = int((x11 + x22) / 2)
        center_y = int((y11 + y22) / 2)
        if Pr_w < 3 or Pr_h < 3:
            continue
        # Pr_img = cv2.cvtColor(Pr_img, cv2.COLOR_BGR2RGB)
        Pr_img = Image.fromarray(Pr_img)

        out = self_model(Gt_img, Pr_img)

        # Square
        dx, dy, ratio, C = out[0], out[1], out[2], out[3]
        # cv2.rectangle(imgc, (x11, y11), (x22, y22), (255, 0, 0), 2)

        if C > 0:
            # Predict
            pr_x = int(center_x - (dx * 128 - 64))
            pr_y = int(center_y - (dy * 128 - 64))
            pr_R = int((length / 2) / (ratio * 0.6 + 0.7))
            # cv2.rectangle(imgc, (pr_x - pr_R, pr_y - pr_R), (pr_x + pr_R, pr_y + pr_R), (0, 0, 255), 1, 4)
            cv2.putText(imgc, 'PR', (pr_x - pr_R, pr_y - pr_R + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),
                        thickness=1)
        PR_END = img_o[pr_y - pr_R:pr_y + pr_R, pr_x - pr_R:pr_x + pr_R, :]

    plt.title(im_id)
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.subplot(1, 4, 2)
    plt.imshow(Gt_img)
    plt.subplot(1, 4, 3)
    plt.imshow(density)

    plt.subplot(1, 4, 4)
    plt.imshow(bina_density, cmap="gray")

    plt.show()

    cv2.imwrite('./bina_density/' + im_id, bina_density)
    return len(contours), density
    # count_s,box_s

    cv2.imwrite('./bina_density/' + im_id, bina_density)
    return len(contours), density
    # count_s,box_s


def isRectangleOverlap(rec1, rec2) -> bool:
    # [x1, y1, x2, y2]
    if rec1[0] > rec2[2] or rec1[1] > rec2[3] or rec2[0] > rec1[2] or rec2[1] > rec1[3]:
        return False
    else:
        return True


def calculate_IoU(predicted_bound, ground_truth_bound):
    pxmin, pymin, pxmax, pymax = predicted_bound
    gxmin, gymin, gxmax, gymax = ground_truth_bound

    parea = (pxmax - pxmin) * (pymax - pymin)
    garea = (gxmax - gxmin) * (gymax - gymin)

    xmin = max(pxmin, gxmin)
    ymin = max(pymin, gymin)
    xmax = min(pxmax, gxmax)
    ymax = min(pymax, gymax)

    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return 0

    area = w * h

    IoU = area / (parea + garea - area)
    return IoU


def siamese_rects(img, density_list, ref_img, rects, coco, target_file, name, result):
    density = None

    for i in range(len(density_list)):

        density_i = density_list[i]
        density_i = torch.squeeze(density_i, axis=0)
        density_i = torch.squeeze(density_i, axis=0)
        density_i = density_i.cpu().numpy() * 50000
        density_i = density_i.astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        if i == 0:
            density = cv2.morphologyEx(density_i, cv2.MORPH_OPEN, kernel)
        else:
            density = np.add(density, cv2.morphologyEx(density_i, cv2.MORPH_OPEN, kernel))

    img = np.array(img)
    img = img.astype(np.uint8)
    img_ori = copy.deepcopy(img)
    ref_img = np.array(ref_img)
    ref_img = ref_img.astype(np.uint8)
    ref_img_ori = copy.deepcopy(ref_img)
    rett, bina_density = cv2.threshold(density, np.max(density) * MIN_DENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)  # 0.4
    contours, hierarchy = cv2.findContours(bina_density, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    width = img.shape[1]
    hight = img.shape[0]
    ref_width = ref_img.shape[1]
    ref_hight = ref_img.shape[0]

    num_rects = len(rects)
    w, h = 0, 0
    Gt_center_ratio = []
    for point in rects:
        y1, x1, y2, x2 = point
        cv2.rectangle(ref_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        w += x2 - x1
        h += y2 - y1
        box_ratio = np.abs(np.abs(y2 - y1) - np.abs(x2 - x1))
        Gt_center_ratio.append(box_ratio)
    max_index = Gt_center_ratio.index(min(Gt_center_ratio))

    w, h = int(w / num_rects), int(h / num_rects)
    Gt_center_point = [int((rects[max_index][3] + rects[max_index][1]) / 2),
                       int((rects[max_index][2] + rects[max_index][0]) / 2)]
    length_x = w * SIAMESE_MAGNIFICATION  # 放大倍数
    length_y = h * SIAMESE_MAGNIFICATION
    length = max(length_x, length_y)
    Gt_x1 = max(int(Gt_center_point[0] - length_x / 2), 0)
    Gt_x2 = min(int(Gt_center_point[0] + length_x / 2), ref_width)
    Gt_y1 = max(int(Gt_center_point[1] - length_y / 2), 0)
    Gt_y2 = min(int(Gt_center_point[1] + length_y / 2), ref_hight)
    # cv2.rectangle(img, (Gt_x1, Gt_y1), (Gt_x2, Gt_y2), (255, 255, 255), 2)
    Gt_point = [Gt_x1, Gt_x2, Gt_y1, Gt_y2]

    centers = []
    density_rects = []
    pr_list = []
    for c in contours:
        # cv2.drawContours(img, c, -1, (0, 0, 255), 2)
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        img = cv2.circle(img, (cx, cy), 2, (0, 0, 255), 4)

        centers.append([cx, cy])

        min_rect = cv2.minAreaRect(c)
        box = np.int64(cv2.boxPoints(min_rect))

        side = max(w, h)
        box_size = max(
            max(box[0][0], box[1][0], box[2][0], box[3][0]) - min(box[0][0], box[1][0], box[2][0], box[3][0]),
            (max(box[0][1], box[1][1], box[2][1], box[3][1]) - min(box[0][1], box[1][1], box[2][1], box[3][1])))

        x1 = max(int(min(box[0][0], box[1][0], box[2][0], box[3][0]) - (side + box_size) / 2 * QUERY_PADDING_RATIO), 0)
        y1 = max(int(min(box[0][1], box[1][1], box[2][1], box[3][1]) - (side + box_size) / 2 * QUERY_PADDING_RATIO), 0)
        x2 = min(int(max(box[0][0], box[1][0], box[2][0], box[3][0]) + (side + box_size) / 2 * QUERY_PADDING_RATIO),
                 width)
        y2 = min(int(max(box[0][1], box[1][1], box[2][1], box[3][1]) + (side + box_size) / 2 * QUERY_PADDING_RATIO),
                 hight)
        if (x2 - x1) < NOISE_SIZE or (y2 - y1) < NOISE_SIZE:
            continue

        extract_w = x2 - x1
        extract_h = y2 - y1
        extract_cx = int((x1 + x2) / 2)
        extract_cy = int((y1 + y2) / 2)
        extract_x1 = max(extract_cx - int(extract_w * EXTRACT_MAFNIFICATION / 2), 0)
        extract_x2 = min(extract_cx + int(extract_w * EXTRACT_MAFNIFICATION / 2), width)
        extract_y1 = max(extract_cy - int(extract_h * EXTRACT_MAFNIFICATION / 2), 0)
        extract_y2 = min(extract_cy + int(extract_h * EXTRACT_MAFNIFICATION / 2), hight)

        # cv2.polylines(img, [box], True, (25, 25, 255), 3)
        cv2.rectangle(img, (extract_x1, extract_y1), (extract_x2, extract_y2), (255, 0, 0), 4)

        num_rects = len(density_rects)

        if num_rects == 1:
            density_rects.append([x1, x2, y1, y2])

        else:

            merge = False

            for i in range(0, num_rects):
                rect_x1 = density_rects[i][0]
                rect_x2 = density_rects[i][1]
                rect_y1 = density_rects[i][2]
                rect_y2 = density_rects[i][3]

                rect_cx = int((rect_x1 + rect_x2) / 2)
                rect_cy = int((rect_y1 + rect_y2) / 2)

                if (abs(rect_cx - cx) + abs(rect_cy - cy)) < MERGE_DISTANCE:
                    density_rects[i][0] = min(rect_x1, x1)
                    density_rects[i][1] = max(rect_x2, x2)
                    density_rects[i][2] = min(rect_y1, y1)
                    density_rects[i][3] = max(rect_y2, y2)
                    merge = True
                    break

            if merge == False:
                density_rects.append([x1, x2, y1, y2])

    all_merged = False
    while not all_merged:

        num_density_rects = len(density_rects)
        all_merged = True

        for i in range(0, num_density_rects):
            if i < (num_density_rects - 1):
                i_x1, i_x2, i_y1, i_y2 = density_rects[i]
                i_cx = int((i_x1 + i_x2) / 2)
                i_cy = int((i_y1 + i_y2) / 2)
                i_diagonal = int(math.sqrt((i_x2 - i_x1) * (i_x2 - i_x1) + (i_y2 - i_y1) * (i_y2 - i_y1)))

                for j in range(i + 1, num_density_rects):
                    j_x1, j_x2, j_y1, j_y2 = density_rects[j]
                    j_cx = int((j_x1 + j_x2) / 2)
                    j_cy = int((j_y1 + j_y2) / 2)
                    j_diagonal = int(math.sqrt((j_x2 - j_x1) * (j_x2 - j_x1) + (j_y2 - j_y1) * (j_y2 - j_y1)))
                    distance = int(math.sqrt((i_cx - j_cx) * (i_cx - j_cx) + (i_cy - j_cy) * (i_cy - j_cy)))

                    if isRectangleOverlap([i_x1, i_y1, i_x2, i_y2], [j_x1, j_y1, j_x2, j_y2]) or distance < int(
                            (i_diagonal + j_diagonal) * MERGE_RATIO):
                        i_x1 = min(i_x1, j_x1)
                        i_y1 = min(i_y1, j_y1)
                        i_x2 = max(i_x2, j_x2)
                        i_y2 = max(i_y2, j_y2)
                        density_rects[i] = i_x1, i_x2, i_y1, i_y2
                        del density_rects[j]
                        all_merged = False
                        break

            if not all_merged:
                break

    count = 0
    ori_rects = copy.deepcopy(density_rects)
    for rect in density_rects:
        x1, x2, y1, y2 = rect
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        d = max(abs(x2 - x1), abs(y2 - y1))

        x1 = max(cx - int(d / 2), 0)
        x2 = min(cx + int(d / 2), width)
        y1 = max(cy - int(d / 2), 0)
        y2 = min(cy + int(d / 2), hight)

        density_rects[count] = x1, x2, y1, y2
        count += 1

        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Predict
    Gt_img_pri = ref_img[Gt_y1:Gt_y2, Gt_x1:Gt_x2, :]
    # Gt_img = cv2.cvtColor(Gt_img, cv2.COLOR_BGR2RGB)
    Gt_img_pri = Image.fromarray(Gt_img_pri)

    Gt_img = ref_img_ori[Gt_y1:Gt_y2, Gt_x1:Gt_x2, :]
    Gt_img = Image.fromarray(Gt_img)

    img_c = copy.deepcopy(img_ori)
    img_result = copy.deepcopy(img_ori)

    index = 0
    for Pr_point in density_rects:

        x11, x22, y11, y22 = Pr_point
        Pr_img = img_ori[y11:y22, x11:x22, :]
        Pr_w, Pr_h = Pr_img.shape[0], Pr_img.shape[1]

        if Pr_w < NOISE_SIZE or Pr_h < NOISE_SIZE:
            continue

        center_x = int((x11 + x22) / 2)
        center_y = int((y11 + y22) / 2)

        Pr_img = Image.fromarray(Pr_img)
        out = self_model(Gt_img, Pr_img)

        dx, dy, dw, dh, C = out[0], out[1], out[2], out[3], out[4]

        for gt_rect in coco:
            cv2.rectangle(img_c, (gt_rect[0], gt_rect[1]), (gt_rect[2], gt_rect[3]), (255, 255, 0), 4, 4)

        if C > CONFIDENCE_THRESHOLD:

            pr_x = int(center_x - (dx * 128 - 64))
            pr_y = int(center_y - (dy * 128 - 64))
            pr_w = int(length * dw / 2)
            pr_h = int(length * dh / 2)
            pr_x1 = max((pr_x - pr_w), 0)
            pr_y1 = max((pr_y - pr_h), 0)
            pr_x2 = min((pr_x + pr_w), width)
            pr_y2 = min((pr_y + pr_h), hight)
            cv2.rectangle(img_c, (pr_x1, pr_y1), (pr_x2, pr_y2), (0, 0, 255), 4, 4)
            cv2.rectangle(img_result, (pr_x1, pr_y1), (pr_x2, pr_y2), (0, 0, 255), 4, 4)

            # if (pr_x2 - pr_x1) > (length * IGNORE_RATIO) and (pr_y2 - pr_y1) > (length * IGNORE_RATIO):

            pr_list.append([pr_x1, pr_y1, pr_x2, pr_y2])

        else:
            pr_x1 = ori_rects[index][0]
            pr_x2 = ori_rects[index][1]
            pr_y1 = ori_rects[index][2]
            pr_y2 = ori_rects[index][3]

            cv2.rectangle(img_c, (pr_x1, pr_y1), (pr_x2, pr_y2), (0, 0, 255), 4, 4)
            cv2.rectangle(img_result, (pr_x1, pr_y1), (pr_x2, pr_y2), (0, 0, 255), 4, 4)

            # if (pr_x2 - pr_x1) > (length * IGNORE_RATIO) and (pr_y2 - pr_y1) > (length * IGNORE_RATIO):

            pr_list.append([pr_x1, pr_y1, pr_x2, pr_y2])

        index += 1

    max_iou = 0.0

    for pr_rect in pr_list:
        for gt_rect in coco:
            iou = abs(calculate_IoU(pr_rect, gt_rect))
            if max_iou < iou:
                max_iou = iou

    plt.subplot(2, 3, 1)
    plt.imshow(Gt_img_pri)
    plt.axis('off')
    plt.title('Support Image ({:d} shot)'.format(len(density_list)), fontsize=7)

    plt.subplot(2, 3, 2)
    plt.imshow(img_ori)
    plt.axis('off')
    plt.title('Query Image', fontsize=7)

    plt.subplot(2, 3, 3)
    plt.imshow(density)
    plt.axis('off')
    plt.title('Density Map', fontsize=7)

    plt.subplot(2, 3, 4)
    img2 = 0.2989 * img_ori[:, :, 0] + 0.5870 * img_ori[:, :, 1] + 0.1140 * img_ori[:, :, 2]
    plt.imshow(img2, cmap='gray')
    plt.imshow(density, cmap=plt.cm.viridis, alpha=0.5)
    plt.axis('off')
    plt.title('Overlaid Density Map', fontsize=7)

    plt.subplot(2, 3, 5)
    plt.imshow(img)
    plt.axis('off')
    plt.title('The Top Possible Regions after\nPurification and NMS Process', fontsize=7)

    plt.subplot(2, 3, 6)
    plt.imshow(img_c)
    plt.axis('off')
    plt.title('SDM-RAN Result: IOU={:.2f}'.format(max_iou), fontsize=7)

    plt.savefig(result, dpi=300)

    # img_save = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(result, img_save)

    if PRINT:
        plt.show()

    elif PRINT == False and max_iou > 0.5:
        temp = cv2.imread(target_file + name)
        cv2.imwrite('./choosen/' + name, temp)

    return pr_list


def format_for_plotting(tensor):

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()
