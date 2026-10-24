"""05 · 训练 ResNet-18 on CIFAR-10"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from resnet_model import ResNet18

CONFIG=None

device="cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR=Path(__file__).resolve().parent.parent/"data"
RESULTS=Path(__file__).resolve().parent/"results.csv"

def parse_args():
    p=argparse.ArgumentParser(
        description="训练 ResNet-18 on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    p.add_argument("--epochs",type=int,default=40,help="训练轮数")
    p.add_argument("--lr",type=float,default=1e-3,help="初始学习率（余弦退火到0）")
    p.add_argument("--batch-size",type=int,default=128,help="训练batch大小")
    p.add_argument("--weight-decay",type=float,default=0.0,help="权重衰减，AdamW建议1e-2量级")
    p.add_argument("--workers",type=int,default=4,help="DataLoader子进程数，Windows需__main__保护")
    p.add_argument("--seed",type=int,default=0,help="随机种子")
    p.add_argument("--optim",choices=["adam","adamw"],default="adam",help="优化器类型")
    p.add_argument("--no-augment",action="store_true",help="关闭数据增强")
    p.add_argument("--erase",type=float,default=0.5,help="RandomErasing的概率，0表示关闭")
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
        "weight_decay": args.weight_decay,
        "optim": args.optim,
        "no_augment": args.no_augment,
        "erase": args.erase,
        "use_bn": True,
        "use_cosine": True,
        "resume":args.resume,
        "tag":args.tag,
    }

def build_transforms(cfg):
    """根据配置拼出训练集和测试集的transform"""
    ops=[]
    if not cfg["no_augment"]:
        ops.append(transforms.RandomCrop(32,padding=4))
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if not cfg["no_augment"] and cfg["erase"] >0:
        ops.append(transforms.RandomErasing(p=cfg["erase"],scale=(0.02,0.2)))
    return transforms.Compose(ops)

def aug_name(cfg):
    """自动生成增强标签，避免手写漏更新"""
    if cfg["no_augment"]:
        return "none"
    return "crop+flip"+("+erase" if cfg["erase"] >0 else "")

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

def main():
    print("="*50,flush=True)
    for k,v in CONFIG.items():
        print(f" {k:15s} = {v}",flush=True)
    print("="*50,flush=True)

    torch.manual_seed(CONFIG["seed"])
    print(f"device={device} workers={CONFIG['num_workers']}",flush=True)

    ckpt_path = CKPT_DIR / f"resnet18_{CONFIG['tag']}.pth"
    print(f"checkpoint -> {ckpt_path.name}",flush=True)

    train_transform=build_transforms(CONFIG)
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
    print(f"optimizer={CONFIG['optim']} wd={CONFIG['weight_decay']}",flush=True)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}",flush=True)

    criterion=nn.CrossEntropyLoss()
    OptClass=torch.optim.AdamW if CONFIG["optim"]=="adamw" else torch.optim.Adam
    optimizer=OptClass(model.parameters(),lr=CONFIG["lr"],weight_decay=CONFIG["weight_decay"])
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
            keys = ("epochs", "batch_size", "lr", "weight_decay",
                    "optim", "no_augment", "erase")
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
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epochs", "batch", "lr", "workers", "augment", "bn", "cosine", "weight_decay", "model", "train_acc", "test_acc"])
        w.writerow([CONFIG["epochs"], CONFIG["batch_size"], CONFIG["lr"], CONFIG["num_workers"],
                    aug_name(CONFIG), CONFIG["use_bn"], CONFIG["use_cosine"],CONFIG["weight_decay"],
                    "ResNet18",f"{train_acc:.4f}", f"{test_acc:.4f}"])
    print(f"结果已追加到 {RESULTS.name}")

if __name__=="__main__":
    CONFIG=build_config(parse_args())
    main()