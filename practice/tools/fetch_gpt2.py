"""把 GPT-2 权重和分词器文件下到本地目录，供 `from_pretrained(本地路径)` 完全离线加载。

为什么不用 `from_pretrained("gpt2")`：
    这台机器的链路会随机掐断长连接——pip 卡在 CloseWait 十几分钟、
    torchvision 下 ViT 权重下到 ~60% 被截断并报 hash 不匹配、HF API 请求 20 s 超时。
    所以下载必须自己做三件事：**断点续传 / 失败重试 / 下完校验完整性**。

镜像顺序：huggingface.co 直连优先，失败退 hf-mirror.com。

用法：
    python practice/tools/fetch_gpt2.py            # 下载
    python practice/tools/fetch_gpt2.py --check    # 只检查本地文件是否齐全
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO = "gpt2"
MIRRORS = ["https://huggingface.co", "https://hf-mirror.com"]
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEST = ROOT / ".cache" / "hf" / REPO

# (文件名, 是否必需, 允许缺失时的替代文件)
FILES: list[tuple[str, bool]] = [
    ("config.json", True),
    ("generation_config.json", False),
    ("tokenizer_config.json", False),
    ("vocab.json", False),
    ("merges.txt", False),
    ("tokenizer.json", False),
]
WEIGHT_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) gpt2-fetch/1.0"
CHUNK = 1 << 20  # 1 MB


def _url(mirror: str, fname: str) -> str:
    return f"{mirror}/{REPO}/resolve/main/{fname}"


def download(fname: str, dest: pathlib.Path, retries: int = 8) -> bool:
    """断点续传下载单个文件。返回是否成功。"""
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        mirror = MIRRORS[(attempt - 1) % len(MIRRORS)]
        url = _url(mirror, fname)
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                # 服务器不认 Range 时返回 200，必须从头写
                if have and r.status != 206:
                    have = 0
                total = int(r.headers.get("Content-Length") or 0) + have
                t0 = time.time()
                next_mark = 32 << 20
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if have >= next_mark:
                            sp = have / 1024 / 1024 / max(time.time() - t0, 1e-6)
                            print(f"      {have / 1024 / 1024:7.1f} / {total / 1024 / 1024:.1f} MB"
                                  f"   {sp:5.2f} MB/s", flush=True)
                            next_mark += 32 << 20

            done = part.stat().st_size
            if total and done != total:
                print(f"      不完整 {done / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB，重试")
                continue
            part.replace(dest)
            print(f"      OK  {done / 1024 / 1024:.2f} MB  ({fname})", flush=True)
            return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"      404 {fname} @ {mirror}")
                break  # 这个文件不存在，换下一个候选
            print(f"      第 {attempt} 次失败 HTTP {e.code}，重试")
        except Exception as e:
            print(f"      第 {attempt} 次失败 {type(e).__name__}: {str(e)[:60]}，重试", flush=True)

        time.sleep(min(2 ** attempt, 15))

    if part.exists() and part.stat().st_size == 0:
        part.unlink(missing_ok=True)
    return False


def check() -> bool:
    if not DEST.exists():
        print(f"目录不存在：{DEST}")
        return False
    ok = True
    for fname, required in FILES:
        p = DEST / fname
        if p.exists():
            print(f"  ✅ {fname:26} {p.stat().st_size / 1024:9.1f} KB")
        elif required:
            print(f"  ❌ {fname:26} 缺失（必需）")
            ok = False
        else:
            print(f"  ·  {fname:26} 缺失（可选）")
    for w in WEIGHT_CANDIDATES:
        p = DEST / w
        if p.exists():
            print(f"  ✅ {w:26} {p.stat().st_size / 1024 / 1024:9.1f} MB")
            break
    else:
        print(f"  ❌ 权重文件缺失（{' 或 '.join(WEIGHT_CANDIDATES)}）")
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        return 0 if check() else 1

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"目标目录：{DEST}\n")

    for fname, required in FILES:
        dest = DEST / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  已有 {fname}")
            continue
        print(f"  下载 {fname}")
        if not download(fname, dest) and required:
            print(f"\n必需文件 {fname} 失败，停止。")
            return 1

    # 权重：按候选顺序试，第一个成功的就够
    if any((DEST / w).exists() for w in WEIGHT_CANDIDATES):
        print("  已有权重文件")
    else:
        for w in WEIGHT_CANDIDATES:
            print(f"  下载权重 {w}（约 550 MB）")
            if download(w, DEST / w):
                break
        else:
            print("\n权重全部失败。")
            return 1

    print()
    ok = check()
    if ok:
        print(f"\n下一步：\n  from transformers import GPT2LMHeadModel\n"
              f"  model = GPT2LMHeadModel.from_pretrained(r'{DEST}', local_files_only=True)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
