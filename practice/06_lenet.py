import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

CONFIG=None

device="cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR=Path(__file__).resolve().parent.parent/"data"
RESULTS=Path(__file__).resolve().parent/"results.csv"

def parse_args():
    p=argparse.ArgumentParser(
        description="训练 LeNet5 on MINIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    p.add_argument("--epochs",type=int,default=40,help="训练轮数")
    p.add_argument("--lr",type=float,default=1e-3,help="初始学习率（余弦退火到0）")
    p.add_argument("--batch-size",type=int,default=128,help="训练batch大小")
    p.add_argument("--workers",type=int,default=4,help="DataLoader子进程数，Windows需__main__保护")
    p.add_argument("--seed",type=int,default=0,help="随机种子")
    p.add_argument("--optim",choices=["adam","adamw"],default="adam",help="优化器类型")
    p.add_argument("--resume",action="store_true",help="从对应 tag 的 checkpoint 接着训练")
    p.add_argument("--tag",type=str,default="default",help="实验标签，决定 checkpoint 文件名")
    return p.parse_args()

def build_config(args):
    return {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "num_workers": args.workers,
        "optim": args.optim,
        "resume":args.resume,
        "tag":args.tag,
    }

def build_transforms(cfg):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
    ])

def evaluate(model,loader):
    model.eval()
    correct=total=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            correct+=(model(xb).argmax(dim=1)==yb).sum().item()
            total+=yb.size(0)
    return correct/total

CKPT_DIR=Path(__file__).resolve().parent.parent/"checkpoints"

def save_ckpt(path,model,optimizer,scheduler,epoch,best_acc,config):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "epoch":epoch,
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),
        "best_acc":best_acc,
        "config":config,
    },path)



class LeNet5(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,6,5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6,16,5,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,120,5,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(),

        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(120*1*1,84),
            nn.ReLU(),
            nn.Linear(84,num_classes),
        )

    def forward(self,x):
        x=self.features(x)
        return self.classifier(x)

def main():
    print("="*50,flush=True)
    for k,v in CONFIG.items():
        print(f" {k:15s} = {v}",flush=True)
    print("="*50,flush=True)

    torch.manual_seed(CONFIG["seed"])
    print(f"device={device} workers={CONFIG['num_workers']}",flush=True)

    ckpt_path = CKPT_DIR / f"lenet_{CONFIG['tag']}.pth"
    print(f"checkpoint -> {ckpt_path.name}",flush=True)

    train_transform=build_transforms(CONFIG)
    test_transform=build_transforms(CONFIG)

    train_set = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], persistent_workers=(CONFIG["num_workers"] > 0),
                              pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=CONFIG["num_workers"], persistent_workers=(CONFIG["num_workers"] > 0),
                             pin_memory=(device == "cuda"))

    model=LeNet5().to(device)
    print(f"optimizer={CONFIG['optim']}",flush=True)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}",flush=True)

    criterion=nn.CrossEntropyLoss()

    OptClass=torch.optim.AdamW if CONFIG["optim"]=="adamw" else torch.optim.Adam
    optimizer=OptClass(model.parameters(),lr=CONFIG["lr"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG["epochs"])

    start_epoch = 1
    best_acc = 0.0

    if CONFIG["resume"]:
        if not ckpt_path.exists():
            print(f"[resume] 没找到 {ckpt_path.name}，从头开始训练", flush=True)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            old = ckpt.get("config", {})

            # 检查 1：影响训练轨迹的超参必须一致
            keys = ("epochs", "batch_size", "lr", "optim")
            bad = [f"{k}: 存档={old.get(k)!r}  当前={CONFIG.get(k)!r}"
                   for k in keys
                   if k in old and old.get(k) != CONFIG.get(k)]
            if bad:
                print("[resume] 配置和存档不一致，拒绝续跑：", flush=True)
                for b in bad:
                    print("         " + b, flush=True)
                raise SystemExit(f"  -> 想重新训练请先删除 {ckpt_path}")

            # 检查 2：余弦调度的 T_max 必须一致
            old_tmax = ckpt.get("scheduler", {}).get("T_max")
            if old_tmax is not None and old_tmax != CONFIG["epochs"]:
                raise SystemExit(
                    f"  -> scheduler 的 T_max={old_tmax}，当前 --epochs={CONFIG['epochs']}；"
                    "续跑时 epochs 必须和原来相同，否则 lr 曲线会跳变"
                )

            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_acc = ckpt.get("best_acc", 0.0)
            print(f"[resume] 从第 {start_epoch} 轮继续"
                  f"（存档 epoch={ckpt['epoch']}，best={best_acc * 100:.2f}%）", flush=True)

    if start_epoch > CONFIG["epochs"]:
        print(f"[resume] 已训练到第 {start_epoch - 1} 轮，"
              f"而 --epochs={CONFIG['epochs']}，无需继续", flush=True)
        return

    for epoch in range(start_epoch,CONFIG["epochs"]+1):
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
        # 这一轮结束
        best_acc=max(best_acc,test_acc)
        save_ckpt(ckpt_path,model,optimizer,scheduler,epoch,best_acc,CONFIG)

    new_file=not RESULTS.exists()
    from tools.cost_lookup import COST_COLS,cost_columns
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epochs", "batch", "lr", "workers", "augment", "bn", "cosine", "weight_decay", "model", "train_acc", "test_acc"]+COST_COLS)
        w.writerow([CONFIG["epochs"], CONFIG["batch_size"], CONFIG["lr"], CONFIG["num_workers"],
                    "norm", True, True, 0.0, "LeNet5-MNIST",
                    f"{train_acc:.4f}", f"{test_acc:.4f}"]+cost_columns("LeNet5-MNIST"))
    print(f"结果已追加到 {RESULTS.name}")

if __name__=="__main__":
    CONFIG=build_config(parse_args())
    main()

