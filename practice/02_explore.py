import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

device="cuda" if torch.cuda.is_available() else "cpu"
print(f"设备:{device}")

DATA_DIR=Path(__file__).resolve().parent.parent/"data"

transform=transforms.ToTensor()

train_set=datasets.CIFAR10(root=str(DATA_DIR),train=True,download=True,transform=transform)

test_set=datasets.CIFAR10(root=str(DATA_DIR),train=False,download=True,transform=transform)

print(f"训练集：{len(train_set)}张")
print(f"测试集：{len(test_set)}张")

# img,label=train_set[0]
#
# print(f"shape : {tuple(img.shape)}")
# print(f"dtype : {img.dtype}")
# print(f"min   : {img.min().item():.4f}")
# print(f"max   : {img.max().item():.4f}")
# print(f"label : {label}")
#
# print(f"类别名: {train_set.classes}")
# print(f"标签 6 对应: {train_set.classes[6]}")

# counts = Counter(train_set.targets)
# print(f"\n{'idx':>3}  {'类别':<12} {'数量':>6}")
# for idx, name in enumerate(train_set.classes):
#     print(f"{idx:3d}  {name:<12} {counts[idx]:6d}")

# fig,axes=plt.subplots(2,4,figsize=(12,6))
# for i,ax in enumerate(axes.flat):
#     img,label=train_set[i]
#     ax.imshow(img.permute(1,2,0))
#     ax.set_title(train_set.classes[label])
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

train_loader=DataLoader(train_set,batch_size=64,shuffle=True)
test_set=DataLoader(test_set,batch_size=256,shuffle=False)

xb,yb=next(iter(train_loader))
print(f"一个batch:x={tuple(xb.shape)}  y={tuple(yb.shape)}")
print(f"标签:{yb.tolist()}")

class SmallCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,num_classes),
        )

    def forward(self,x):
        return self.classifier(self.features(x))

model=SmallCNN().to(device)
print(f"参数量:{sum(p.numel() for p in model.parameters()):,}")

with torch.no_grad():
    out=model(xb.to(device))
print(f"输入{tuple(xb.shape)}->输出{tuple(out.shape)}")