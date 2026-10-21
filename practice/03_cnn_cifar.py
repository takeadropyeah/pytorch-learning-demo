"""
03.小CNN打CIFAR-10

复用 01_baseline的训练循环骨架，只改三处：
    1.损失函数 MSELoss      ->CrossEntropyLoss
    2.评估指标 loss         ->准确率（argmax）
    3.数据    合成回归数据    ->CIFAR-10
    相对上一步的变化：
    4. 加 __main__ 保护（Windows 上 num_workers>0 的硬性要求）
    5. DataLoader 加 num_workers / persistent_workers / pin_memory
    6. 记录每轮耗时，并把最终结果追加到 results.csv
"""

import  csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

CONFIG = {
    "seed": 0,
    "batch_size": 128,
    "lr": 1e-3,
    "epochs": 40,
    "num_workers":4,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent / "results.csv"

# torch.manual_seed(CONFIG["seed"])
#
# train_transform=transforms.Compose([
#     transforms.RandomCrop(32,padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
#
# test_transform=transforms.ToTensor()
#
# train_set=datasets.CIFAR10(root=str(DATA_DIR),train=True,download=True,transform=train_transform)
# test_set=datasets.CIFAR10(root=str(DATA_DIR),train=False,download=True,transform=test_transform)
#
# train_loader=DataLoader(train_set,batch_size=CONFIG["batch_size"],shuffle=True)
# test_loader=DataLoader(test_set,batch_size=256,shuffle=False)

class SmallCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),             #新增
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),             #新增
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

# model=SmallCNN().to(device)
# criterion=nn.CrossEntropyLoss()
# optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG["lr"])

def evaluate(model,loader):
    model.eval()
    correct=total=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            pred=model(xb).argmax(dim=1)
            correct+=(pred==yb).sum().item()
            total+=yb.size(0)
        return correct/total

def main():
    torch.manual_seed(CONFIG["seed"])
    print(f"device={device}  workers={CONFIG["num_workers"]}",flush=True)

    train_transform=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform=transforms.ToTensor()

    train_set=datasets.CIFAR10(root=str(DATA_DIR),train=True,download=True,transform=train_transform)
    test_set=datasets.CIFAR10(root=str(DATA_DIR),train=False,download=True,transform=test_transform)

    train_loader=DataLoader(
        train_set,batch_size=CONFIG["batch_size"],shuffle=True,
        num_workers=CONFIG["num_workers"],persistent_workers=True,
        pin_memory=(device=="cuda"),
    )
    test_loader=DataLoader(
        test_set,batch_size=256,shuffle=False,
        num_workers=CONFIG["num_workers"],persistent_workers=True,
        pin_memory=(device=="cuda"),
    )

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG["epochs"])

    for epoch in range(1,CONFIG["epochs"]+1):
        t0=time.perf_counter()
        model.train()
        running=0.0
        for i,(xb,yb) in enumerate(train_loader):
            xb=xb.to(device,non_blocking=True)
            yb=yb.to(device,non_blocking=True)

            loss=criterion(model(xb),yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running+=loss.item()
            if(i+1)%100==0:
                print(f"epoch{epoch}{i+1}/{len(train_loader)} loss={running/100:.4f}",flush=True)
                running=0.0

        train_acc=evaluate(model,train_loader)
        test_acc=evaluate(model,test_loader)
        scheduler.step()
        print(f"==>epoch{epoch} lr={optimizer.param_groups[0]['lr']:.6f} "
              f"train{train_acc*100:.2f}% test{test_acc*100:.2f}% "
              f"耗时{time.perf_counter()-t0:.1f}s",flush=True)

    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epochs", "batch", "lr", "workers", "augment", "train_acc", "test_acc"])
        w.writerow([CONFIG["epochs"], CONFIG["batch_size"], CONFIG["lr"],
                        CONFIG["num_workers"], "crop+flip",
                        f"{train_acc:.4f}", f"{test_acc:.4f}"])
    print(f"结果已追加到 {RESULTS.name}")


if __name__=="__main__":
    main()


# for epoch in range(1,CONFIG["epochs"]+1):
#     model.train()
#     running=0.0
#     for i,(xb,yb) in enumerate(train_loader):
#         xb,yb=xb.to(device),yb.to(device)
#
#         loss=criterion(model(xb),yb)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running+=loss.item()
#         if(i+1)%100==0:
#             print(f"epoch{epoch} {i+1}/{len(train_loader)} loss={running/100:.4f}",flush=True)
#             running=0.0
#
#     print(f"==>epoch{epoch} train {evaluate(train_loader)*100:.2f}%"
#           f"test {evaluate(test_loader)*100:.2f}%",flush=True)

