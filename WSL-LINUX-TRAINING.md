# WSL Linux 练手清单（10 周 × 30 分钟）

> 目的：进实验室第一周不被环境卡住。实验室几乎全是「Linux 服务器 + SSH + 命令行」，和你现在的 Windows 图形界面是两套习惯。
>
> 你在 Windows 那边学的 PyTorch 一点不用丢——这份清单只练**操作系统的用法**，不涉及模型。
>
> 节奏：每周 1 小时以内，可以和 12 周学习计划并行，不抢认知带宽。

---

## 零、先验证 WSL 是活的

打开 **PowerShell**（不是 WSL），执行：

```powershell
wsl -l -v
```

正常应该看到类似：

```
  NAME      STATE           VERSION
* Ubuntu    Stopped         2
```

然后：

```powershell
wsl
```

你会直接进入 Ubuntu 的 shell，提示符变成 `用户名@机器名:~$`。

如果报错：

| 报错 | 处理 |
|---|---|
| `Wsl/E_ACCESSDENIED` | 用非管理员 PowerShell 重试；关闭杀软/EDR 后重试 |
| 提示需要启用虚拟机平台 | 管理员 PowerShell 跑 `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart` 然后重启 |
| 启动很慢或卡住 | `wsl --update` 更新 WSL 到最新版 |

⚠️ 顺带说一句：**Linux 区分大小写，Windows 不区分**。`Data/` 和 `data/` 是两个不同的目录。这是 Windows 用户进 Linux 第一周最容易踩的坑，比任何命令都常见。

---

## 一、十条练习（每条 20–40 分钟）

### 练习 1 · 认识你在哪

```bash
pwd                      # 我在哪个目录
whoami                   # 我是谁
cat /etc/os-release      # 什么系统
ls -la ~                 # 家目录里有什么
df -h                    # 磁盘还剩多少
```

再看一个关键路径：

```bash
ls /mnt/                 # 你的 Windows 盘都挂在这里
ls /mnt/d/pytorch-learning/
```

**完成的标志**：你能说清三件事——你的家目录绝对路径、Windows 的 `D:\pytorch-learning` 在 Linux 里的路径、`~` 代表什么。

**实验室对应**：连上服务器第一句话就是 `pwd` 和 `df -h`（先看磁盘满没满，很多"训练莫名失败"是磁盘满了）。

---

### 练习 2 · 文件操作

```bash
mkdir -p ~/proj/code ~/proj/logs ~/proj/results   # -p 一次建多层
cd ~/proj && pwd
cp -r /mnt/d/pytorch-learning/examples ~/proj/code/
ls -lh ~/proj/code/examples
du -sh ~/proj                          # 看目录总大小
find ~/proj -name "*.py" | head        # 按名字找文件
```

**完成的标志**：`examples` 下的 5 个 `.py` 已经复制到 `~/proj/code/examples`，且 `du -sh` 能看到占用。

**实验室对应**：找数据、清理旧输出、腾出磁盘空间。

---

### 练习 3 · 看日志（最高频操作）

先造一个日志：

```bash
printf 'INFO start\nINFO epoch 1 loss=2.31\nERROR cuda out of memory\nINFO epoch 2 loss=1.92\n' > ~/proj/logs/demo.log
```

然后：

```bash
head -2 ~/proj/logs/demo.log          # 头两行
tail -2 ~/proj/logs/demo.log          # 尾两行
wc -l ~/proj/logs/demo.log            # 多少行
grep ERROR ~/proj/logs/demo.log       # 找错误
grep -c INFO ~/proj/logs/demo.log     # 数出现次数
grep -n loss ~/proj/logs/demo.log     # 带行号
less ~/proj/logs/demo.log             # 翻页看，q 退出
```

`tail -f` 是核心命令（实时盯着日志滚动），但要用练习 4 的后台任务配合才看得到效果。

**完成的标志**：你能用一条命令回答"这个日志里有没有 OOM，在哪一行"。

**实验室对应**：90% 的排查从看日志开始。

---

### 练习 4 · 后台运行 —— 最重要的一课

**先理解为什么需要它**：你在笔记本上跑训练，合上盖子/断网/关终端，进程就被杀了。这是新手最常见的"我明明跑了 8 小时结果全没了"。

先写一个假训练脚本（**故意用假脚本**，不下载任何东西）：

```bash
cat > ~/proj/fake_train.sh << 'EOF'
#!/usr/bin/env bash
for i in $(seq 1 120); do
    loss=$(awk "BEGIN{printf \"%.4f\", 2.0/(1+$i/10)}")
    echo "[$(date '+%H:%M:%S')] step $i/120 loss=$loss"
    sleep 1
done
echo "training done"
EOF
```

然后：

```bash
chmod +x ~/proj/fake_train.sh                    # 加执行权限
cd ~/proj
nohup ./fake_train.sh > logs/train.log 2>&1 &    # 后台跑，输出重定向
echo "进程号: $!"
```

现在验证它真的在跑：

```bash
tail -f logs/train.log       # 实时看，Ctrl+C 退出（不会杀掉训练）
ps aux | grep fake_train     # 确认进程还活着
```

**关键实验**：`Ctrl+C` 退出 `tail` → 关掉整个 WSL 终端 → 重新打开 WSL → 再 `tail -f logs/train.log`。**日志还在长**，说明训练没死。

结束时清理：

```bash
kill <刚才的进程号>
```

**完成的标志**：你亲手体验过"关掉终端训练继续跑"。

**实验室对应**：这是每一天都在用的东西。

---

### 练习 5 · 重定向与管道的含义

```bash
echo hello > /tmp/a.txt          # 覆盖写
echo world >> /tmp/a.txt         # 追加
cat /tmp/a.txt | wc -l           # 管道：前一个的输出给后一个
ls /nonexistent 2> /tmp/err.txt  # 只重定向错误
cat /tmp/err.txt
ls /nonexistent > /tmp/out.txt 2>&1   # 正确+错误一起重定向
```

理解那个 `2>&1`：`1` 是标准输出，`2` 是标准错误，`2>&1` 意思是"把错误也扔进标准输出"。`nohup ... > log 2>&1 &` 里的含义就是"标准输出和错误都写进 log，然后放后台"。

**完成的标志**：能解释 `nohup ./run.sh > out.log 2>&1 &` 里每一段的含义。

---

### 练习 6 · tmux（比 nohup 更好用）

`nohup` 能让训练不中断，但你**回不去那个终端**做交互。`tmux` 两者都能。

只学五个操作，够用一年：

| 操作 | 命令 |
|---|---|
| 新建会话 | `tmux new -s train` |
| 分离（回到普通 shell） | 按 `Ctrl+b`，松开，再按 `d` |
| 查看会话列表 | `tmux ls` |
| 重连 | `tmux attach -t train` |
| 杀掉会话 | `tmux kill-session -t train` |

**任务**：`tmux new -s train` → 在里面跑 `~/proj/fake_train.sh` → `Ctrl+b d` 分离 → 关掉 WSL 终端 → 重开 → `tmux attach -t train`，你会看到它还在跑。

**完成的标志**：能闭着眼完成"新建 → 跑任务 → 分离 → 重连"。

**实验室对应**：服务器上所有人都在 tmux 里干活，因为 SSH 会断。

---

### 练习 7 · 环境管理

你的 WSL 里应该本来就装了 Anaconda。先确认：

```bash
which python
python --version
conda --version
conda env list
```

然后练一遍完整的环境操作：

```bash
conda create -n t1 python=3.11 -y
conda activate t1
which python                 # ← 关键：确认指向 t1 环境，不是 base
pip install numpy -q
python -c "import numpy; print(numpy.__version__)"
conda deactivate
```

**完成的标志**：你能用 `which python` 判断"我现在在哪个环境里"，并且能说清 `conda activate` 到底改了什么。

**实验室对应**：一台服务器多个项目共用，环境混用是最常见的"代码在我电脑上能跑"事故来源。

---

### 练习 8 · SSH（先写好配置，等有服务器就上手）

没有服务器也能练：先在本机装 SSH 服务，然后连自己。

```bash
sudo apt update && sudo apt install -y openssh-server
sudo service ssh start
ssh-keygen -t ed25519 -C "你的邮箱"    # 一路回车
cat ~/.ssh/id_ed25519.pub              # 这是公钥，可以给别人
```

然后 `ssh $(whoami)@localhost`，首次会问 `yes/no`，输入 `yes`，再输 Linux 用户密码。

再配置一个别名（这是你以后用得最多的东西）：

```bash
cat > ~/.ssh/config << 'EOF'
Host lab
    HostName 192.168.1.100        # 换成真实服务器地址
    User yourname
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60        # 每 60 秒发心跳，防止闲置被踢
EOF
chmod 600 ~/.ssh/config
```

以后只要 `ssh lab` 就进去了。

**完成的标志**：`ssh $(whoami)@localhost` 能连上自己；`~/.ssh/config` 写好了模板。

**实验室对应**：每天第一件事就是 SSH 进服务器。

---

### 练习 9 · 传文件（rsync 比 scp 好得多）

```bash
# 先演练，加 --dry-run 不会真写
rsync -avz --dry-run --progress \
  /mnt/d/pytorch-learning/examples/ \
  ~/proj/code/examples/

# 去掉 --dry-run 真正同步
rsync -avz --progress \
  --exclude '__pycache__/' \
  --exclude 'data/' \
  /mnt/d/pytorch-learning/examples/ \
  ~/proj/code/examples/
```

**注意末尾那个斜杠**：`examples/` 是"把里面的内容同步过去"，`examples` 是"把 examples 这个目录本身同步过去"。这是 rsync 最经典的坑。

`rsync` 相比 `scp` 的三个优势：断点续传、只传变化的文件、支持排除。所以实验室里大家基本都用 rsync。

**完成的标志**：第二次同步时，你会看到它跳过了没变的文件（输出里没有新传输）。

---

### 练习 10 · 串成一次"服务器工作流"

把前面全串起来，模拟真实工作流：

```bash
tmux new -s work                      # 1. 进会话
cd ~/proj && ./fake_train.sh > logs/run1.log 2>&1   # 2. 跑任务
# Ctrl+b d  分离，关掉终端，再重新打开 wsl

tmux attach -t work                   # 3. 回来看
tail -f logs/run1.log                 # 4. 盯进度
grep -c "step" logs/run1.log          # 5. 数跑到第几步了
grep "done" logs/run1.log             # 6. 确认结束了没有
cp logs/run1.log ~/proj/results/      # 7. 归档结果
ls -lh ~/proj/results/
```

**完成的标志**：全程不用鼠标、不用图形界面，靠命令完成"启动 → 离开 → 回来 → 检查 → 归档"。

**实验室对应**：这就是实验室里一天的节奏。

---

## 二、有服务器之后再加练的

| 主题 | 命令 | 说明 |
|---|---|---|
| 排队抢卡 | `sinfo`、`sbatch`、`squeue -u $USER`、`scancel <id>` | 大组标配 slurm |
| 看 GPU 占用 | `nvidia-smi`、`watch -n 2 nvidia-smi` | 谁在占卡一目了然 |
| 磁盘配额 | `quota -s`、`df -h` | 别把公共盘写满 |
| 进程管理 | `ps aux`、`top`、`htop`、`kill -9` | 清理自己的僵尸进程 |

一个 slurm 脚本的样子（等用到了再细看）：

```bash
#!/bin/bash
#SBATCH --job-name=myexp
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.log

python train.py --config configs/resnet18.yaml
```

提交：`sbatch train.slurm`；看队列：`squeue -u $USER`。

---

## 三、四个必踩的坑

**1. 换行符（CRLF）**

你用 Windows 侧的工具编辑了 `.sh`，在 Linux 里会报：

```
bash: ./run.sh: /usr/bin/env bash^M: bad interpreter
```

那个 `^M` 就是 Windows 的回车。修复：

```bash
sed -i 's/\r$//' run.sh
# 或者
dos2unix run.sh
```

**2. `/mnt/d` 很慢**

跨文件系统访问，IO 可能慢 5–10 倍。所以：**代码和日志放 `~/` 里，大数据集可以放 `/mnt/d/`**。在 `/mnt/d` 上直接训练会明显变慢，还会拖垮文件监视。

**3. WSL 默认吃一半物理内存**

你现在没配 `.wslconfig`，意味着 WSL 最多能用宿主一半内存。训练时可能把 Windows 拖卡。在 `C:\Users\Lenovo\.wslconfig` 里写：

```ini
[wsl2]
memory=8GB
processors=8
swap=4GB
```

改完 `wsl --shutdown` 再启动才生效。

**4. WSL 磁盘只增不减**

`ext4.vhdx` 现在 22 GB，删了文件也不会自动缩小。需要手动压缩（先 `wsl --shutdown`）：

```powershell
Optimize-VHD -Path 'D:\WSL\Ubuntu\ext4.vhdx' -Mode Full
```

`Optimize-VHD` 需要 Hyper-V 模块，没有的话用 `diskpart` 的 `compact vdisk`。

---

## 四、进度打勾

- [ ] 0. `wsl -l -v` 正常，能进入 Ubuntu
- [ ] 1. 认清 `pwd` / `~` / `/mnt/d` / 大小写敏感
- [ ] 2. `mkdir -p`、`cp -r`、`du -sh`、`find`
- [ ] 3. `tail` / `grep` / `wc` / `less` 看日志
- [ ] 4. `nohup ... &` 后台跑，关终端不死
- [ ] 5. 能解释 `> log 2>&1 &` 每一段
- [ ] 6. tmux 五件套闭眼能做
- [ ] 7. `which python` 判断当前环境
- [ ] 8. `ssh-keygen` + `~/.ssh/config` 模板
- [ ] 9. `rsync` 同步（会加 `--dry-run` 和 `--exclude`）
- [ ] 10. 完整跑一遍"启动 → 离开 → 回来 → 检查 → 归档"

十条打满，进实验室第一周你就能直接上手干活。
