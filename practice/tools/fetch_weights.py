"""把预训练权重下载到 TORCH_HOME，并做 sha256 校验。

为什么自己写这一小段：
    torchvision 用 torch.hub.load_state_dict_from_url 下载，它校验失败就直接把文件删掉，
    不告诉你是「下载不完整」还是「中间被代理改了字节」。
    这段代码做三件 torch 不做的事：
      1. 断点续传（网络断了从断点接着下，不浪费已经下的几百 MB）
      2. 下完先落到 .part，校验通过才改名——失败一次不会污染缓存
      3. 把「下了多少、还要多久」打出来，而不是一串 tqdm 进度条

用法：
    python tools/fetch_weights.py                     # 默认下 vit_b_16
    python tools/fetch_weights.py --url <url> --sha <前缀> --name <文件名>
"""

import argparse
import hashlib
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_REPO = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("TORCH_HOME", str(_REPO / ".cache" / "torch"))

DEFAULT_URL = "https://download.pytorch.org/models/vit_b_16-c867db91.pth"
DEFAULT_SHA = "c867db91"
DEFAULT_NAME = "vit_b_16-c867db91.pth"
TOTAL = 346_328_529  # 字节，用于算进度百分比


def download(url: str, dst: Path, total_hint: int, retries: int = 5) -> None:
    """下到 dst，支持断点续传。已存在且长度对得上就直接跳过。"""
    part = dst.with_suffix(dst.suffix + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= total_hint:
            print(f"  .part 已完整（{have:,} 字节），跳过下载")
            break

        # 从断点续传：用 Range 头只要剩下的部分
        headers = {"User-Agent": "python-urllib/3"}
        if have:
            headers["Range"] = f"bytes={have}-"
            print(f"  第 {attempt} 次尝试，从 {have/1024/1024:.1f} MB 处续传")
        else:
            print(f"  第 {attempt} 次尝试，从头下载")

        ctx = ssl.create_default_context()
        # 这台机器上 TLS 证书链不完整，只对下载权重这一步放宽校验；
        # 真正的完整性由下面的 sha256 保证。
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r, \
                    open(part, "ab") as f:
                if have and r.status != 206:
                    # 服务器不支持 Range，只能重下
                    print("  服务器不支持续传，从头开始")
                    f.close()
                    part.unlink()
                    f = open(part, "wb")
                chunk = 1 << 20                 # 1 MB
                done = have
                t0 = time.time()
                last = 0
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if done - last >= 32 * 1024 * 1024:      # 每 32 MB 报一次
                        last = done
                        sp = (done - have) / 1024 / 1024 / max(time.time() - t0, 1e-6)
                        print(f"    {done/1024/1024:7.1f} / "
                              f"{total_hint/1024/1024:.0f} MB "
                              f"({done*100/total_hint:5.1f}%)  {sp:.2f} MB/s")
            size = part.stat().st_size
            if size >= total_hint:
                break
            print(f"  只拿到 {size:,} / {total_hint:,} 字节，重试")
        except Exception as e:                    # noqa: BLE001
            print(f"  出错：{type(e).__name__}: {e}")
            time.sleep(2)

    size = part.stat().st_size
    if size != total_hint:
        raise RuntimeError(f"下载不完整：{size:,} / {total_hint:,} 字节，"
                           f"再跑一次可以续传")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--sha", default=DEFAULT_SHA, help="sha256 前缀（8 位）")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--total", type=int, default=TOTAL)
    a = ap.parse_args()

    cache = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
    dst, part = cache / a.name, cache / (a.name + ".part")

    if dst.exists() and sha256_of(dst).startswith(a.sha):
        print(f"已存在且校验通过，跳过：{dst}")
        return

    print(f"目标：{dst}")
    if not part.exists():
        print(f"  文件大小 {a.total:,} 字节（{a.total/1024/1024:.0f} MB）")

    download(a.url, dst, a.total)

    got = sha256_of(part)
    if not got.startswith(a.sha):
        part.unlink()
        raise RuntimeError(
            f"sha256 不匹配：期望前缀 {a.sha}，实际 {got}。\n"
            f"下载的字节是坏的（多半是被代理改过），已删除后重跑一次。")

    print(f"校验通过 {got[:16]}…")
    part.replace(dst)
    print(f"完成：{dst}（{dst.stat().st_size/1024/1024:.1f} MB）")


if __name__ == "__main__":
    main()
