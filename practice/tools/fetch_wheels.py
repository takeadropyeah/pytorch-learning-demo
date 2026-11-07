"""从镜像站直接下载 wheel 到本地 wheelhouse，绕过 pip 卡死的网络层。

背景（2026-11-06）：这台机器上 `pip install` 会无输出地卡住十几分钟，进程里
到 pypi.tuna 的连接停在 CloseWait（对端已关闭、本端还在等），而同一个镜像用
纯 urllib 取 20 KB 只要 0.2 秒。所以 pip 的 HTTP 栈（urllib3）在这条链路失效，
但 urllib 正常。

做法：本脚本用 urllib 把 wheel 以及它的全部传递依赖下载到
`.pip-tmp/wheels/`，然后：

    .venv\\Scripts\\python.exe -m pip install --no-index \
        --find-links=.pip-tmp/wheels transformers peft

pip 自己不碰网络，只做本地解析和安装。

用法：
    python practice/tools/fetch_wheels.py transformers peft
    python practice/tools/fetch_wheels.py --list          # 只列，不下载
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from email.parser import Parser
from importlib.metadata import distributions

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion

# 阿里云镜像实测 0.2 s 返回索引页；tuna 12.5 s、pypi.org 48.4 s。
INDEX = "http://mirrors.aliyun.com/pypi/simple/{}/"
UA = "pip/26.2.1"

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEST = ROOT / ".pip-tmp" / "wheels"


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def installed_names() -> dict[str, str]:
    out = {}
    for d in distributions():
        name = d.metadata["Name"]
        if name:
            out[canonicalize_name(name)] = d.version
    return out


def pick_wheel(name: str) -> tuple[str, pathlib.Path]:
    """返回 (下载 URL, 文件名)。挑最新的、非预发布的、本平台可用的 wheel。"""
    index_url = INDEX.format(name)
    html = _get(index_url).decode("utf-8", "replace")

    hrefs = re.findall(r'href="([^"]+)"', html)
    best = None
    for href in hrefs:
        # 有些镜像把 #sha256=... 附在链接上
        clean = href.split("#", 1)[0]
        if not clean.endswith(".whl"):
            continue
        fname = clean.rsplit("/", 1)[-1]
        try:
            wname, wver, _build, tags = parse_wheel_filename(fname)
        except Exception:
            continue
        if canonicalize_name(str(wname)) != name:
            continue
        if wver.is_prerelease or wver.is_devrelease:
            continue

        # 只要 pure-python，或 cp313 / abi3 的 win_amd64
        # 注意：parse_wheel_filename 返回的是一组 Tag 对象，字段名是 interpreter/abi/platform
        ok = False
        for tag in tags:
            if tag.interpreter == "py3" and tag.abi == "none" and tag.platform == "any":
                ok = True
                break
            if tag.abi in ("cp313", "abi3", "none") and tag.platform == "win_amd64":
                ok = True
                break
        if not ok:
            continue

        if best is None or wver > best[0]:
            url = urllib.parse.urljoin(index_url, clean)
            best = (wver, url, fname)

    if best is None:
        raise RuntimeError(f"镜像上没有找到 {name} 的可用于 cp313/win_amd64 的 wheel")
    return best[1], pathlib.Path(best[2])


def download(url: str, dest: pathlib.Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    已有  {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    data = _get(url, timeout=120)
    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"    {len(data) / 1024 / 1024:6.2f} MB  {dest.name}")


def requires_of(wheel: pathlib.Path) -> list[str]:
    """从 wheel 里的 METADATA 读 Requires-Dist，过滤掉 extra 与不满足 marker 的。"""
    with zipfile.ZipFile(wheel) as z:
        meta_name = next((n for n in z.namelist() if n.endswith(".dist-info/METADATA")), None)
        if meta_name is None:
            return []
        raw = z.read(meta_name).decode("utf-8", "replace")

    msg = Parser().parsestr(raw)
    out = []
    for line in msg.get_all("Requires-Dist") or []:
        try:
            req = Requirement(line)
        except Exception:
            continue
        if req.marker is not None:
            # 跳过可选 extra，其余按当前环境求值
            marker_str = str(req.marker)
            if "extra" in marker_str:
                continue
            try:
                if not req.marker.evaluate():
                    continue
            except Exception:
                continue
        out.append(canonicalize_name(req.name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="*", default=["transformers", "peft"])
    ap.add_argument("--list", action="store_true", help="只列出会下载什么，不实际下载")
    ap.add_argument("--index", action="store_true", help="为已下载的 wheel 重新展开依赖")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    have = installed_names()

    queue = [canonicalize_name(p) for p in (args.packages or ["transformers", "peft"])]
    seen: set[str] = set()
    plan: list[tuple[str, str, pathlib.Path]] = []
    failed: list[str] = []

    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        if name in have:
            print(f"  [已装] {name} {have[name]}")
            continue
        try:
            url, fname = pick_wheel(name)
        except Exception as e:
            print(f"  [失败] {name}: {e}")
            failed.append(name)
            continue
        plan.append((name, url, fname))
        print(f"  [待下] {name:22} {fname}")

    if args.list:
        print(f"\n共 {len(plan)} 个 wheel 需要下载；加 --index 可展开其依赖。")
        return 0

    print(f"\n下载到 {DEST}")
    download_queue = list(plan)
    expanded: set[str] = set()
    failed: list[str] = []

    # 反复展开，直到没有新的包为止（传递依赖可能有好几层）
    while download_queue:
        batch, download_queue = download_queue, []
        fresh: list[pathlib.Path] = []

        for name, url, fname in batch:
            dest = DEST / fname
            try:
                download(url, dest)
                fresh.append(dest)
            except Exception as e:
                print(f"  [失败] {name}: {type(e).__name__}: {e}")
                failed.append(name)

        for w in fresh:
            if w.name in expanded:
                continue
            expanded.add(w.name)
            try:
                deps = requires_of(w)
            except Exception as e:
                print(f"  [跳过] {w.name}: {e}")
                continue
            for d in deps:
                if d in seen or d in have:
                    continue
                seen.add(d)
                try:
                    durl, dfname = pick_wheel(d)
                except Exception as e:
                    print(f"  [失败] {d}: {e}")
                    failed.append(d)
                    continue
                download_queue.append((d, durl, dfname))

    total = sum(p.stat().st_size for p in DEST.glob("*.whl"))
    print(f"\nwheelhouse 共 {len(list(DEST.glob('*.whl')))} 个文件，{total / 1024 / 1024:.1f} MB")
    if failed:
        print(f"\n未解决：{failed}")
        print("可手动补：python practice/tools/fetch_wheels.py <包名>")
    print("\n下一步：")
    print("  .venv\\Scripts\\python.exe -m pip install --no-index "
          f"--find-links={DEST} transformers peft")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
