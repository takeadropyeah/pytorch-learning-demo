import argparse

parser = argparse.ArgumentParser(description="argparse 最小示例")

parser.add_argument("--epochs",type=int,default=40,help="训练轮数")
parser.add_argument("--lr",type=float,default=1e-3,help="学习率")
parser.add_argument("--no-augment",action="store_true",help="关闭数据增强")

args = parser.parse_args()

print(f"epochs      ={args.epochs}")
print(f"lr          ={args.lr}")
print(f"no_augment  ={args.no_augment}")