"""
给训练脚本补齐 results.csv 的成本三列（params / macs_g / infer_ms）。

为什么单独一个文件
------------------
这三个数由 `tools/model_stats.py` 测出来，写在 `tools/costs.json` 里。
训练脚本每跑完一次就查一下这张表，把成本列一起写进 csv——
这样 results.csv 的列数是稳定的，不会出现「表头 14 列、新行 11 列」的错位。

如果 costs.json 还不存在（比如换了一台机器），就写空字符串，
之后跑一次 `python .\\tools\\model_stats.py --fill` 补上。
"""

import json
from pathlib import Path

COST_COLS = ["params", "macs_g", "infer_ms"]
_COSTS_PATH = Path(__file__).resolve().parent / "costs.json"


def cost_columns(model_name):
    """返回 [params, macs_g, infer_ms]；查不到就给空字符串。"""
    if not _COSTS_PATH.exists():
        return [""] * len(COST_COLS)
    try:
        data = json.loads(_COSTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [""] * len(COST_COLS)

    entry = data.get(model_name, {})
    return [entry.get(k, "") for k in COST_COLS]
