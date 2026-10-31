"""05 · 训练 ResNet-18 on CIFAR-10"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from augment_mix import apply_mix, mix_criterion
from resnet_model import build_resnet

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
    p.add_argument("--no-normalize",action="store_true",help="关闭 Normalize（复现 10-27 及之前的历史实验时才需要）")
    p.add_argument("--erase",type=float,default=0.5,help="RandomErasing的概率，0表示关闭")
    p.add_argument("--resume",action="store_true",help="从对应 tag 的 checkpoint 接着训练")
    p.add_argument("--plain",action="store_true",help="去掉残差短接，退化成 plain-18（消融用）")
    p.add_argument("--layers",type=int,default=18,choices=[18,34],help="ResNet 深度：18 或 34")
    p.add_argument("--mix",choices=["none","cutmix","mixup"],default="none",help="batch 级混合增强（none 时行为与原来完全一致）")
    p.add_argument("--mix-alpha",type=float,default=1.0,help="混合的 Beta(α,α) 参数，α=1 退化成均匀分布")
    p.add_argument("--mix-prob",type=float,default=0.5,help="每个 batch 施加混合增强的概率（论文用 0.5）")
    p.add_argument("--no-lam-recompute",action="store_true",help="[对照实验] CutMix 不做实际面积重算，直接用 Beta 采样的 lam 当标签权重")
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
        "normalize": not args.no_normalize,
        "erase": args.erase,
        "use_bn": True,
        "use_cosine": True,
        "plain": args.plain,
        "layers":args.layers,
        "mix": args.mix,
        "mix_alpha": args.mix_alpha,
        "mix_prob": args.mix_prob,
        "recompute_lam": not args.no_lam_recompute,
        "resume":args.resume,
        "tag":args.tag,
    }

# CIFAR-10 官方统计量：把整个训练集 50000 张、每张 32×32 的像素按通道累加求出来的，
# 不是猜的。换数据集必须重新算，直接用这两个数会得到错误的分布。
CIFAR_MEAN=(0.4914,0.4822,0.4465)
CIFAR_STD =(0.2023,0.1994,0.2010)

def build_transforms(cfg,train=True):
    """根据配置拼出 transform

    train=True  : 增强 + ToTensor + Normalize
    train=False : ToTensor + Normalize

    注意：test 分支**绝不能**带 RandomCrop/Flip/Erasing——那是标签破坏，
    会让"测试集准确率"这个数字失去意义。
    """
    ops=[]
    if train and not cfg["no_augment"]:
        ops.append(transforms.RandomCrop(32,padding=4))
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    # RandomErasing 保持在 ToTensor 之后、Normalize 之前：
    # torchvision 文档的推荐顺序是 Normalize 再 Erasing，这里故意不跟，
    # 目的是让 P5 只改 Normalize 一个变量，不和 erasing 的位置混在一起。
    if train and not cfg["no_augment"] and cfg["erase"] >0:
        ops.append(transforms.RandomErasing(p=cfg["erase"],scale=(0.02,0.2)))
    if cfg.get("normalize",True):
        ops.append(transforms.Normalize(CIFAR_MEAN,CIFAR_STD))
    return transforms.Compose(ops)

def aug_name(cfg):
    """自动生成增强标签，避免手写漏更新"""
    if cfg["no_augment"]:
        base = "none"
    else:
        base = "crop+flip"+("+erase" if cfg["erase"] >0 else "")
    if cfg.get("normalize",True):
        base += "+norm"
    if cfg.get("mix", "none") != "none":
        base += f"+{cfg['mix']}(a={cfg['mix_alpha']:g},p={cfg.get('mix_prob',0.5):g})"
        if cfg.get("mix") == "cutmix" and not cfg.get("recompute_lam", True):
            base += "-norc"
    return base

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

    ckpt_path = CKPT_DIR / f"resnet{CONFIG['layers']}_{CONFIG['tag']}.pth"
    print(f"checkpoint -> {ckpt_path.name}",flush=True)

    train_transform=build_transforms(CONFIG,train=True)
    test_transform=build_transforms(CONFIG,train=False)

    train_set = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], persistent_workers=True,
                              pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=CONFIG["num_workers"], persistent_workers=True,
                             pin_memory=(device == "cuda"))

    model=build_resnet(str(CONFIG["layers"]),plain=CONFIG["plain"]).to(device)
    print(f"optimizer={CONFIG['optim']} wd={CONFIG['weight_decay']}",flush=True)
    print(
        f"结构: {'plain' if CONFIG['plain'] else 'ResNet'}-{CONFIG['layers']}（{'无' if CONFIG['plain'] else '有'}残差）",
        flush=True)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}",flush=True)

    criterion=nn.CrossEntropyLoss()
    print(f"混合增强: {CONFIG['mix']}"
          + (f"（Beta({CONFIG['mix_alpha']:g},{CONFIG['mix_alpha']:g})）" if CONFIG['mix']!='none' else "（关闭，行为与原来一致）"),
          flush=True)
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
                    "optim", "no_augment", "normalize", "erase", "plain", "mix", "mix_alpha", "mix_prob",
                    "recompute_lam","layers")
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

            # 混合增强：mix='none' 时原样返回（lam=1、y_a==y_b==yb），走下面同一条 loss
            if CONFIG["mix"]!="none" and torch.rand(1).item()<CONFIG["mix_prob"]:
                xb,ya,yb2,lam=apply_mix(CONFIG["mix"],xb,yb,CONFIG["mix_alpha"])
                loss=mix_criterion(model(xb),ya,yb2,lam)
            else:
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
    model_name = f"{'Plain' if CONFIG['plain'] else 'ResNet'}{CONFIG['layers']}"
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epochs", "batch", "lr", "workers", "augment", "bn", "cosine", "weight_decay", "model", "train_acc", "test_acc"]+COST_COLS)
        w.writerow([CONFIG["epochs"], CONFIG["batch_size"], CONFIG["lr"], CONFIG["num_workers"],
                    aug_name(CONFIG), CONFIG["use_bn"], CONFIG["use_cosine"],CONFIG["weight_decay"],
                    model_name,
                    f"{train_acc:.4f}", f"{test_acc:.4f}"]+cost_columns(model_name))
    print(f"结果已追加到 {RESULTS.name}")

if __name__=="__main__":
    CONFIG=build_config(parse_args())
    main()