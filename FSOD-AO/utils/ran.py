import time
from PIL import Image
from torchvision import transforms

import torch
import torch.nn as nn


#from dataset import lable_loader,test_lable_loader

model = torch.load("./data/models/Siamese.pt", map_location=torch.device("cpu"))
model.eval()

def mmodel(path1,path2):
    ref = Image.open(path1)
    img = Image.open(path2)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(0, 1),
    ])

    image = transform(img)
    ref = transform(ref)
    image = torch.unsqueeze(image, 0)
    ref = torch.unsqueeze(ref, 0)

    with torch.no_grad():
        start = time.time()
        output = model(ref, image)
        end = time.time()
        print(end-start)
    return output.numpy()[0]

def self_model(ref, img):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(0, 1),
    ])

    image = transform(img)
    ref = transform(ref)
    image = torch.unsqueeze(image, 0)
    ref = torch.unsqueeze(ref, 0)

    with torch.no_grad():
        output = model(ref, image)
    return output.numpy()[0]



class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
        return

    def forward(self, t1, t2):
        loss1 = nn.MSELoss(reduction='sum')(t1, t2)
        return loss1


