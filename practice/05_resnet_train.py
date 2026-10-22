"""05 · 训练 ResNet-18 on CIFAR-10"""

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from  torchvision import datasets,transforms

from resnet_model import ResNet18

CONFIG={
    "seed":0,
    "batch_size":128,
    "lr":1e-3,
    "epochs":40,
    "num_workers":4,
    "weight_decay":1e-4,
    "use_bn":True,
    "use_cosine":True,
}

device="cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR=Path(__file__).resolve().parent.parent/"data"
RESULTS=Path(__file__).resolve().parent/"results.csv"

def evaluate(model,loader):
    model.eval()
    correct=total=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            correct+=(model(xb).argmax(dim=1)==yb).sum().item()
            total+=yb.size(0)
    return correct/total

def main():
    torch.manual_seed(CONFIG["seed"])
    print(f"device={device} workers={CONFIG["num_workers"]}",flush=True)

    train_transform=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform=transforms.ToTensor()

    train_set = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], persistent_workers=True,
                              pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=CONFIG["num_workers"], persistent_workers=True,
                             pin_memory=(device == "cuda"))

    model=ResNet18().to(device)
    print(f"optimizer=AdamW wd={CONFIG["weight_decay"]}",flush=True)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}",flush=True)

    criterion=nn.CrossEntropyLoss()
    # optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG["lr"])
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=CONFIG["lr"],weight_decay=CONFIG["weight_decay"]
    )
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
                print(f"epoch {epoch} {i+1}/{len(train_loader)} loss={running/100:.4f}",flush=True)
                running=0.0

        train_acc=evaluate(model,train_loader)
        test_acc=evaluate(model,test_loader)
        print(f"==>epoch {epoch} lr={optimizer.param_groups[0]['lr']:.6f} "
              f"train {train_acc*100:.2f}% test {test_acc*100:.2f}% "
              f"耗时{time.perf_counter()-t0:.1f}s",flush=True)
        scheduler.step()

    new_file=not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epochs", "batch", "lr", "workers", "augment", "bn", "cosine", "weight_decay", "model", "train_acc", "test_acc"])
        w.writerow([CONFIG["epochs"], CONFIG["batch_size"], CONFIG["lr"], CONFIG["num_workers"],
                    "crop+flip", CONFIG["use_bn"], CONFIG["use_cosine"],CONFIG["weight_decay"],
                    "ResNet18",f"{train_acc:.4f}", f"{test_acc:.4f}"])
    print(f"结果已追加到 {RESULTS.name}")

if __name__=="__main__":
    main()