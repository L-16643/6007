"""OBD 常量、下载、读取、清洗与商品特征对齐。

原始数据布局（全量版与官方样例一致）::

    <root>/<policy>/<campaign>/<campaign>.csv
    <root>/<policy>/<campaign>/item_context.csv

每行是一次推荐位曝光：``timestamp, item_id, position (1..3), click,
propensity_score, user_feature_0..3, user-item_affinity_0..K``。
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
CAMPAIGNS = ("all", "men", "women")
POLICIES = ("random", "bts")
N_ITEMS = {"all": 80, "men": 34, "women": 46}
LEN_LIST = 3  # 每次曝光 3 个推荐位

USER_CAT_COLS = [f"user_feature_{k}" for k in range(4)]
ITEM_NUM_COLS = ["item_feature_0"]
ITEM_CAT_COLS = [f"item_feature_{k}" for k in range(1, 4)]

FULL_DATASET_URL = "https://research.zozo.com/data_release/open_bandit_dataset.zip"
SAMPLE_BASE_URL = "https://raw.githubusercontent.com/st-tech/zr-obp/master/obd"


def affinity_cols(campaign: str) -> list[str]:
    return [f"user-item_affinity_{k}" for k in range(N_ITEMS[campaign])]


# ----------------------------------------------------------------------------
# 下载
# ----------------------------------------------------------------------------
def _remote_size(url: str) -> int | None:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size = r.headers.get("Content-Length")
    return int(size) if size else None


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """支持断点续传（HTTP Range）的下载。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total is not None and have == total:
        print(f"[skip] {dest} 已完整 ({total / 1e6:.1f} MB)")
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
        if have and r.status != 206:  # 服务器不支持续传 -> 重新下载
            f.seek(0)
            f.truncate()
            have = 0
        done = have
        while block := r.read(chunk):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({done / total:6.1%})", end="", flush=True)
    print()


def download_sample(out: str | Path = "data/sample") -> None:
    """官方 1 万行样例（obp GitHub 仓库，约 30 MB）。"""
    out = Path(out)
    for p in POLICIES:
        for c in CAMPAIGNS:
            for name in (f"{c}.csv", "item_context.csv"):
                dest = out / p / c / name
                if not (dest.exists() and dest.stat().st_size > 0):
                    print(f"下载样例 {p}/{c}/{name}")
                    download_file(f"{SAMPLE_BASE_URL}/{p}/{c}/{name}", dest)


def download_full(out: str | Path = "data/raw") -> None:
    """全量 OBD（约 413 MB zip，2690 万行），下载并解压到 ``out``。"""
    out = Path(out)
    zpath = out / "open_bandit_dataset.zip"
    print(f"下载 {FULL_DATASET_URL} -> {zpath}")
    download_file(FULL_DATASET_URL, zpath)
    marker = out / ".extracted"
    if marker.exists():
        print("[skip] 已解压")
        return
    print("解压中 ...")
    with zipfile.ZipFile(zpath) as z:
        if bad := z.testzip():
            raise RuntimeError(f"zip 损坏: {bad}，请删除后重新下载")
        for m in z.namelist():  # 防止路径穿越
            if Path(m).is_absolute() or ".." in Path(m).parts:
                raise RuntimeError(f"zip 中存在不安全路径: {m}")
        z.extractall(out)
    marker.touch()


# ----------------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------------
def find_campaign_dir(root: str | Path, policy: str, campaign: str) -> Path:
    """在 ``root`` 下定位 ``<policy>/<campaign>/``（允许多一层目录，如 zip 内的
    ``open_bandit_dataset/``）。"""
    root = Path(root)
    direct = root / policy / campaign
    if (direct / f"{campaign}.csv").exists():
        return direct
    hits = sorted(root.glob(f"*/{policy}/{campaign}/{campaign}.csv"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(
        f"在 {root} 下找不到 {policy}/{campaign}/{campaign}.csv，请先运行 scripts/prepare_data.py"
    )


def read_logs(root: str | Path, policy: str, campaign: str, max_rows: int | None = None) -> pd.DataFrame:
    """用紧凑 dtype 读取一个日志文件。``max_rows`` 取时间上最早的若干行。"""
    path = find_campaign_dir(root, policy, campaign) / f"{campaign}.csv"
    dtypes = {"item_id": "int16", "position": "int8", "click": "int8", "propensity_score": "float64"}
    dtypes.update({c: "string" for c in USER_CAT_COLS})
    dtypes.update({c: "float32" for c in affinity_cols(campaign)})  # 小整数计数，float32 省一半内存
    if max_rows is None:
        df = pd.read_csv(path, dtype=dtypes, engine="pyarrow")
    else:
        df = pd.read_csv(path, dtype=dtypes, nrows=max_rows)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""])
    for c in USER_CAT_COLS:
        df[c] = df[c].astype("category")
    log.info("读取 %s/%s: %d 行", policy, campaign, len(df))
    return df


def read_item_context(root: str | Path, policy: str, campaign: str) -> pd.DataFrame:
    df = pd.read_csv(find_campaign_dir(root, policy, campaign) / "item_context.csv", index_col=0)
    return df.sort_values("item_id").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 清洗
# ----------------------------------------------------------------------------
def clean_logs(df: pd.DataFrame, campaign: str) -> tuple[pd.DataFrame, dict]:
    """校验并清洗日志。

    * 删除关键列/特征缺失行、完全重复行
    * 保留 ``item_id ∈ [0, n_items)``、``position ∈ {1,2,3}``、``click ∈ {0,1}``、
      ``propensity_score ∈ (0, 1]``、亲和度非负
    * ``position`` 转为 0 起始（OBP 约定），按时间戳稳定排序

    倾向得分**不在这里裁剪**——裁剪是估计器层面的选择，留给敏感度实验。
    """
    n_items = N_ITEMS[campaign]
    report: dict = {"n_raw": int(len(df))}
    key_cols = ["timestamp", "item_id", "position", "click", "propensity_score"]
    feat_cols = USER_CAT_COLS + affinity_cols(campaign)
    if missing := [c for c in key_cols + feat_cols if c not in df.columns]:
        raise ValueError(f"缺少列: {missing[:5]}...")

    mask_na = df[key_cols + feat_cols].isna().any(axis=1)
    report["dropped_missing"] = int(mask_na.sum())
    df = df.loc[~mask_na]
    dup = df.duplicated()
    report["dropped_duplicate"] = int(dup.sum())
    df = df.loc[~dup]

    checks = {
        "dropped_bad_item": ~df["item_id"].between(0, n_items - 1),
        "dropped_bad_position": ~df["position"].between(1, LEN_LIST),
        "dropped_bad_click": ~df["click"].isin([0, 1]),
        "dropped_bad_pscore": ~((df["propensity_score"] > 0) & (df["propensity_score"] <= 1)),
        "dropped_negative_affinity": (df[affinity_cols(campaign)] < 0).any(axis=1),
    }
    bad = np.zeros(len(df), dtype=bool)
    for name, m in checks.items():
        m = m.to_numpy()
        report[name] = int((m & ~bad).sum())
        bad |= m
    df = df.loc[~bad].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["position"] = (df["position"] - 1).astype("int8")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    report.update(
        n_clean=int(len(df)),
        n_clicks=int(df["click"].sum()),
        ctr=float(df["click"].mean()) if len(df) else float("nan"),
        time_start=str(df["timestamp"].iloc[0]) if len(df) else None,
        time_end=str(df["timestamp"].iloc[-1]) if len(df) else None,
        pscore=propensity_summary(df["propensity_score"].to_numpy()),
        ctr_by_position={int(k): float(v) for k, v in df.groupby("position")["click"].mean().items()},
        n_unique_items_shown=int(df["item_id"].nunique()),
    )
    return df, report


def propensity_summary(ps: np.ndarray) -> dict:
    """行为倾向得分分布（1/ps 是重要性权重可能的最大放大倍数）。"""
    if len(ps) == 0:
        return {}
    q = np.quantile(ps, [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0])
    keys = ["min", "q0.1%", "q1%", "q5%", "q25%", "median", "q75%", "max"]
    out = {k: float(v) for k, v in zip(keys, q)}
    out.update(
        mean=float(ps.mean()),
        n_unique=int(np.unique(ps).size),
        max_inverse=float(1.0 / q[0]),
        **{"frac_below_1e-3": float((ps < 1e-3).mean())},
    )
    return out


# ----------------------------------------------------------------------------
# 商品特征对齐
# ----------------------------------------------------------------------------
def align_item_context(canonical: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """OBD 中 random 与 bts 两个文件夹的商品**类别特征哈希盐不同**（数值特征
    ``item_feature_0`` 相同）。校验两边对商品的划分是一一对应（双射）后，统一
    使用 ``canonical`` 的标签，使两个策略共用同一个动作特征矩阵。"""
    if not np.array_equal(canonical["item_id"].to_numpy(), other["item_id"].to_numpy()):
        raise ValueError("两个 item_context 的 item_id 不一致")
    for c in ITEM_NUM_COLS:
        if not np.allclose(canonical[c].to_numpy(), other[c].to_numpy()):
            raise ValueError(f"数值商品特征 {c} 在两个策略间不一致")
    for c in ITEM_CAT_COLS:
        pairs = pd.DataFrame({"a": canonical[c].to_numpy(), "b": other[c].to_numpy()})
        if pairs.groupby("a")["b"].nunique().max() != 1 or pairs.groupby("b")["a"].nunique().max() != 1:
            raise ValueError(f"类别商品特征 {c} 不是简单的重新标注")
    return canonical.copy()
