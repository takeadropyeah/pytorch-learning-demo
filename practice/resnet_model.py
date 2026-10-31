import torch
import torch.nn as nn
#深度 -> 每个阶段的block数
ARCHS={"18":(2,2,2,2),"34":(3,4,6,3)}

class BasicBlock(nn.Module):

    def __init__(self, in_ch,out_ch,stride=1,plain=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # plain=True：去掉短接。整个网络退化成 ResNet 论文里的 plain-18，
        # 深度、通道、BN 位置全都不变，唯一变量就是"有没有 + x"。
        if plain:
            self.shortcut = None
        elif stride !=1 or in_ch !=out_ch:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_ch,out_ch,1,stride=stride,bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut=nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is None:
            out = self.relu(out)
        else:
            out = self.relu(out + self.shortcut(x))
        return out

class ResNet(nn.Module):
    def __init__(self,num_classes=10,plain=False,layers=(2,2,2,2)):
        super().__init__()
        self.layers=layers
        self.plain = plain
        self.stem=nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1=self._make_layer(64,64,layers[0],stride=1)
        self.layer2=self._make_layer(64,128,layers[1],stride=2)
        self.layer3=self._make_layer(128,256,layers[2],stride=2)
        self.layer4=self._make_layer(256,512,layers[3],stride=2)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(512,num_classes)

    def _make_layer(self,in_ch,out_ch,num_blocks,stride):
        blocks=[BasicBlock(in_ch,out_ch,stride,plain=self.plain)]
        for _ in range(num_blocks-1):
            blocks.append(BasicBlock(out_ch,out_ch,1,plain=self.plain))
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

def build_resnet(depth="18",num_classes=10,plain=False):
        return ResNet(num_classes,plain,ARCHS[str(depth)])

def ResNet18(num_classes=10,plain=False):
        return build_resnet("18",num_classes,plain)

