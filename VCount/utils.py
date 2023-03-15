import numpy as np
import torch.nn.functional as F
import math
from torchvision import transforms
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
#matplotlib.use('agg')

model = torch.load("./util/RAN.pt",map_location=torch.device("cuda"))
MAPS = ['map3','map4']
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

def select_exemplar_rois(image):
    all_rois = []

    print("Press 'q' or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar, 'space' to save.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n') or key == '\r':
            rect = cv2.selectROI("image", image, False, False)
            x1 = rect[0]
            y1 = rect[1]
            x2 = x1 + rect[2] - 1
            y2 = y1 + rect[3] - 1

            all_rois.append([y1, x1, y2, x2])
            for rect in all_rois:
                y1, x1, y2, x2 = rect
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print("Press q or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar")

    return all_rois

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def PerturbationLoss(output,boxes,sigma=8, use_gpu=True):
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            out = output[:,:,y1:y2,x1:x2]
            GaussKernel = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu: GaussKernel = GaussKernel.cuda()
            Loss += F.mse_loss(out.squeeze(),GaussKernel)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:,:,y1:y2,x1:x2]
        Gauss = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu: GaussKernel = GaussKernel.cuda()
        Loss += F.mse_loss(out.squeeze(),GaussKernel) 
    return Loss


def MincountLoss(output,boxes, use_gpu=True):
    ones = torch.ones(1)
    if use_gpu: ones = ones.cuda()
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:,:,y1:y2,x1:x2].sum()
            if X.item() <= 1:
                Loss += F.mse_loss(X,ones)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:,:,y1:y2,x1:x2].sum()
        if X.item() <= 1:
            Loss += F.mse_loss(X,ones)  
    return Loss
def self_model(ref,img):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 缩放
        transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
        transforms.Normalize(0, 1),  # 标准化均值为0标准差为1
    ])

    image = transform(img)
    ref = transform(ref)
    image = torch.unsqueeze(image, 0)
    ref = torch.unsqueeze(ref, 0)
    ref = ref.to('cuda')
    image = image.to('cuda')
    with torch.no_grad():
        #s = time.time()
        output = model(ref,image)
        #e = time.time()
        #print(e-s)
    return output.cpu().numpy()[0]

def pad_to_size(feat, desire_h, desire_w):
    """ zero-padding a four dim feature matrix: N*C*H*W so that the new Height and Width are the desired ones
        desire_h and desire_w should be largers than the current height and weight
    """

    cur_h = feat.shape[-2]
    cur_w = feat.shape[-1]

    left_pad = (desire_w - cur_w + 1) // 2
    right_pad = (desire_w - cur_w) - left_pad
    top_pad = (desire_h - cur_h + 1) // 2
    bottom_pad =(desire_h - cur_h) - top_pad

    return F.pad(feat, (left_pad, right_pad, top_pad, bottom_pad))


def extract_features(feature_model, image, boxes,feat_map_keys=['map3','map4'], exemplar_scales=[0.9, 1.1]):
    N, M = image.shape[0], boxes.shape[2]
    """
    Getting features for the image N * C * H * W
    """
    Image_features = feature_model(image)
    """
    Getting features for the examples (N*M) * C * h * w
    """
    for ix in range(0,N):
        # boxes = boxes.squeeze(0)
        boxes = boxes[ix][0]
        #print('111111',boxes)
        cnter = 0
        Cnter1 = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)
            if keys == 'map1' or keys == 'map2':
                Scaling = 4.0
            elif keys == 'map3':
                Scaling = 8.0
            elif keys == 'map4':
                Scaling =  16.0
            else:
                Scaling = 32.0
            boxes_scaled = boxes / Scaling
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1 # make the end indices exclusive 
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)            
            box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
            box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]            
            max_h = math.ceil(max(box_hs))
            max_w = math.ceil(max(box_ws))

            for j in range(0,M):
                y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
                y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4]) 
                #print(y1,y2,x1,x2,max_h,max_w)
                #支撑图像特征：examples_features
                #examples_features = np.load('examples_features_0.npy')

                if j == 0:
                    examples_features = image_features[:,:,y1:y2, x1:x2]
                    if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                        #examples_features = pad_to_size(examples_features, max_h, max_w)
                        examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')                    
                else:
                    feat = image_features[:,:,y1:y2, x1:x2]
                    if feat.shape[2] != max_h or feat.shape[3] != max_w:
                        feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                        #feat = pad_to_size(feat, max_h, max_w)
                    examples_features = torch.cat((examples_features,feat),dim=0)

            #np.save('examples_features_0.npy', examples_features)
            """
            Convolving example features over image features
            """
            h, w = examples_features.shape[2], examples_features.shape[3]
            features =    F.conv2d(
                    F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
                    examples_features
                )
            combined = features.permute([1,0,2,3])
            # computing features for scales 0.9 and 1.1
            for scale in exemplar_scales:
                    h1 = math.ceil(h * scale)
                    w1 = math.ceil(w * scale)
                    if h1 < 1: # use original size if scaled size is too small
                        h1 = h
                    if w1 < 1:
                        w1 = w
                    examples_features_scaled = F.interpolate(examples_features, size=(h1,w1),mode='bilinear')  
                    features_scaled =    F.conv2d(F.pad(image_features, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
                    examples_features_scaled)
                    features_scaled = features_scaled.permute([1,0,2,3])
                    combined = torch.cat((combined,features_scaled),dim=1)
            if cnter == 0:
                Combined = 1.0 * combined
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
                Combined = torch.cat((Combined,combined),dim=1)
            cnter += 1
        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat,Combined.unsqueeze(0)),dim=0)
    return All_feat

class resizeImage(object):

    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample


class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
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
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample


Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
Transform = transforms.Compose([resizeImage( MAX_HW)])
TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)
    return denormalized

def scale_and_clip(val, scale_factor, min_val, max_val):
    "Helper function to scale a value and clip it within range"

    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val

def siamese(img, density, rects, im_id, pred_cnt, gt_cnt, split):

    density = torch.squeeze(density, axis=0)
    density = torch.squeeze(density, axis=0)
    density = density.cpu().numpy()*50000
    density_ = density.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    density = cv2.morphologyEx(density_, cv2.MORPH_OPEN, kernel)

    img = np.array(img)
    img = img.astype(np.uint8)
    img_o = img.copy()
    rett, bina_density = cv2.threshold(density, np.max(density)*0.3, 255, cv2.THRESH_BINARY)#0.4
    contours, hierarchy = cv2.findContours(bina_density, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])  # 求x坐标
        cy = int(M['m01'] / M['m00'])  # 求y坐标
        centers.append([cx,cy])
    w,h =0,0
    Gt_center_ratio = []

    for vvv,point in enumerate(rects):
        if vvv==1:
            break
        y1, x1, y2, x2 = point

        w += x2-x1
        h += y2-y1
        box_ratio =np.abs(np.abs(y2-y1)-np.abs(x2-x1))
        Gt_center_ratio.append(box_ratio)
    max_index = Gt_center_ratio.index(min(Gt_center_ratio))

    w, h =int(w/3),(h/3)
    Gt_center_point = [int((rects[max_index][3]+rects[max_index][1])/2),int((rects[max_index][2]+rects[max_index][0])/2)]
    length = max(w,h)*1.1#放大倍数
    Gt_x1 = max(int(Gt_center_point[0] - length / 2),0)
    Gt_x2 = int(Gt_center_point[0] + length / 2)
    Gt_y1 = max(int(Gt_center_point[1] - length / 2),0)
    Gt_y2 = int(Gt_center_point[1] + length / 2)
    Pr_points = []

    for center in centers:
        x1 = int(center[0] - length/2)
        x2 = int(center[0] + length/2)
        y1 = int(center[1] - length/2)
        y2 = int(center[1] + length/2)
        Pr_points.append([x1,x2,y1,y2])

    #预测
    Gt_img = img_o[Gt_y1:Gt_y2,Gt_x1:Gt_x2,:]
    Gt_img = Image.fromarray(Gt_img)
    object_num = 0
    for ii, Pr_point in enumerate(Pr_points):

        x11, x22, y11, y22 = Pr_point
        Pr_img = img_o[y11:y22,x11:x22,:]
        Pr_w,Pr_h = Pr_img.shape[0],Pr_img.shape[1]
        if Pr_w<3 or Pr_h<3:
            continue

        Pr_img =Image.fromarray(Pr_img)
        out = self_model( Gt_img,Pr_img)
        dx, dy, dw, dh, C = out[0], out[1], out[2], out[3],out[4]#矩形
        if C>0.3:
            object_num += 1
            img = cv2.circle(img, (centers[ii][0], centers[ii][1]), 2, (0, 0, 255), 8)  # 画出重心
    if split== 'visulaztion':
        cv2.imwrite('./visualization_results/'+im_id,img)
    return len(contours), density,object_num

def visualize_output_and_save(input_, output, boxes, save_path, figsize=(20, 12), dots=None):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """

    # get the total count
    pred_cnt = output.sum().item()
    boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(0, boxes.shape[0]):
        y1, x1, y2, x2 = int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(boxes[i, 3].item()), int(
            boxes[i, 4].item())
        roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()
        boxes2.append([y1, x1, y2, x2, roi_cnt])

    img1 = format_for_plotting(denormalize(input_))
    output = format_for_plotting(output)

    fig = plt.figure(figsize=figsize)

    # display the input image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(img1)

    for bbox in boxes2:
        y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

    img2 = 0.2989*img1[:,:,0] + 0.5870*img1[:,:,1] + 0.1140*img1[:,:,2]
    ax.imshow(img2, cmap='gray')
    ax.imshow(output, cmap=plt.cm.viridis, alpha=0.5)


    # display the density map
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output)
    # plt.colorbar()

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output)
    for bbox in boxes2:
        y1, x1, y2, x2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')

    fig.colorbar(ret_fig, ax=ax)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()


