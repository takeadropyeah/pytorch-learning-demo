import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, in_ch,out_ch,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride !=1 or in_ch !=out_ch:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_ch,out_ch,1,stride=stride,bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut=nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out

class ResNet18(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1=self._make_layer(64,64,2,stride=1)
        self.layer2=self._make_layer(64,128,2,stride=2)
        self.layer3=self._make_layer(128,256,2,stride=2)
        self.layer4=self._make_layer(256,512,2,stride=2)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(512,num_classes)

    def _make_layer(self,in_ch,out_ch,num_blocks,stride):
        blocks=[BasicBlock(in_ch,out_ch,stride)]
        for _ in range(num_blocks-1):
            blocks.append(BasicBlock(out_ch,out_ch,1))
        return nn.Sequential(*blocks)

    def forward(self,x):
        x=self.stem(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.pool(x)
        x=torch.flatten(x,1)
        return self.fc(x)

