import torch
import torch.nn as nn
#torch.manual_seed(0)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 8000),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(8000, 4096),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(2048, 1000),
        )

        self.fc2 = torch.nn.Sequential(
            # 64*3*3也要改
            torch.nn.Linear(2000, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5),
        )

    def forward(self,input1,input2):
        output1 = self.forwardAB(input1)
        output2 = self.forwardAB(input2)
        input = torch.cat([output1, output2], dim=1)
        output = self.fc2(input)
        return output

    def forwardAB(self, x):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        block4_out = self.block4(block3_out)
        block5_out = self.block5(block4_out)
        # block6_out=self.block6(block5_out)
        res = block5_out.view(block5_out.size(0), -1)
        out = self.block6(res)
        return out