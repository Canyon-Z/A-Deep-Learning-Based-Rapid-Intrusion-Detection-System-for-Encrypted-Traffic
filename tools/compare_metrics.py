"""训练前后指标对比：把新的 checkpoints/model_metrics.json 与重训前的备份逐项对照。

用途
----
本轮重训做了一件影响全局的事：数据划分从「每个类别各自随机切 pcap 文件」改成
「所有类别合并成整体再切」（旧做法会让某一类的 val/test 全部堆在少数几个 pcap 上，
于是 val 准确率虚高、test 准确率掉档）。重训完后需要快速看清 6 个模型的 test 指标
究竟是变好还是变差，这个脚本就是干这个的。

旧值来源：``checkpoints_backup_pre_retrain/model_metrics.json``（重训前备份）
新值来源：``checkpoints/model_metrics.json``

用法
----
    python tools/compare_metrics.py
    python tools/compare_metrics.py --old <旧文件> --new <新文件>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD = ROOT / "checkpoints_backup_pre_retrain" / "model_metrics.json"
DEFAULT_NEW = ROOT / "checkpoints" / "model_metrics.json"

# (字段名, 中文标签, 单位后缀, 越大越好)
FIELDS: list[tuple[str, str, str, bool]] = [
    ("best_val_accuracy", "验证最高准确率", "%", True),
    ("test_accuracy", "测试准确率", "%", True),
    ("detection_rate", "恶意检出率", "%", True),
    ("fnr", "漏报率 FNR", "%", False),
    ("fpr", "误报率 FPR", "%", False),
    ("precision", "精确率", "", True),
    ("recall", "召回率", "", True),
    ("f1_score", "F1", "", True),
    ("auc", "AUC", "", True),
    ("latency_ms_per_sample", "单样本耗时", "ms", False),
]


def parse_number(value: object) -> float | None:
    """把 "87.83%"、"87.83"、87.83 统一成 float；无法解析时返回 None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_metrics(path: Path) -> dict:
    """读取指标文件，文件缺失或损坏时给出清晰错误。"""
    if not path.exists():
        raise SystemExit(f"指标文件不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取 {path}：{exc}") from exc


def format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "%":
        return f"{value:.2f}%"
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"


def format_delta(new_value: float | None, old_value: float | None, higher_is_better: bool, unit: str) -> str:
    """输出带箭头的变化量，并标注方向好坏。"""
    if new_value is None or old_value is None:
        return "—"
    delta = new_value - old_value
    if abs(delta) < (0.005 if unit == "%" else 0.00005):
        return "持平"
    improved = delta > 0 if higher_is_better else delta < 0
    arrow = "↑" if delta > 0 else "↓"
    sign = "+" if delta > 0 else ""
    body = f"{sign}{delta:.2f}%" if unit == "%" else f"{sign}{delta:.4f}"
    return f"{body} {arrow} {'改善' if improved else '变差'}"


def print_model_table(model: str, old_metrics: dict, new_metrics: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {model}")
    print(f"{'=' * 78}")
    print(f"  {'指标':<14}{'旧值':>14}{'新值':>14}   {'变化':<22}")
    print(f"  {'-' * 74}")
    for key, label, unit, higher_is_better in FIELDS:
        old_value = parse_number(old_metrics.get(key))
        new_value = parse_number(new_metrics.get(key))
        delta_text = format_delta(new_value, old_value, higher_is_better, unit)
        print(f"  {label:<14}{format_value(old_value, unit):>14}{format_value(new_value, unit):>14}   {delta_text:<22}")

    old_epochs = old_metrics.get("epochs_trained")
    new_epochs = new_metrics.get("epochs_trained")
    if old_epochs is not None or new_epochs is not None:
        print(f"  {'训练轮数':<14}{str(old_epochs):>14}{str(new_epochs):>14}")


def main() -> int:
    parser = argparse.ArgumentParser(description="对比重训前后的模型指标")
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD, help="旧指标文件（默认：重训前备份）")
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW, help="新指标文件（默认：checkpoints/model_metrics.json）")
    args = parser.parse_args()

    old_all = load_metrics(args.old)
    new_all = load_metrics(args.new)

    print("指标对比：旧 = {0}".format(args.old))
    print("          新 = {0}".format(args.new))

    models = [name for name in new_all if isinstance(new_all[name], dict)]
    if not models:
        raise SystemExit("新指标文件里没有可对比的模型记录。")

    accuracy_deltas: list[float] = []
    detection_deltas: list[float] = []
    fnr_deltas: list[float] = []

    for model in models:
        old_metrics = old_all.get(model) if isinstance(old_all.get(model), dict) else {}
        new_metrics = new_all[model]
        print_model_table(model, old_metrics, new_metrics)

        old_accuracy = parse_number(old_metrics.get("test_accuracy"))
        new_accuracy = parse_number(new_metrics.get("test_accuracy"))
        if old_accuracy is not None and new_accuracy is not None:
            accuracy_deltas.append(new_accuracy - old_accuracy)

        old_detection = parse_number(old_metrics.get("detection_rate"))
        new_detection = parse_number(new_metrics.get("detection_rate"))
        if old_detection is not None and new_detection is not None:
            detection_deltas.append(new_detection - old_detection)

        old_fnr = parse_number(old_metrics.get("fnr"))
        new_fnr = parse_number(new_metrics.get("fnr"))
        if old_fnr is not None and new_fnr is not None:
            fnr_deltas.append(new_fnr - old_fnr)

    def average(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    print(f"\n{'=' * 78}")
    print("  汇总（各模型平均变化）")
    print(f"{'=' * 78}")
    for label, values, higher_is_better in (
        ("测试准确率", accuracy_deltas, True),
        ("恶意检出率", detection_deltas, True),
        ("漏报率 FNR", fnr_deltas, False),
    ):
        mean_delta = average(values)
        if mean_delta is None:
            print(f"  {label}: 无可用数据")
            continue
        arrow = "↑" if mean_delta > 0 else ("↓" if mean_delta < 0 else "→")
        improved = "改善" if (mean_delta > 0) == higher_is_better and abs(mean_delta) > 0.005 else (
            "变差" if abs(mean_delta) > 0.005 else "持平"
        )
        print(f"  {label}: {mean_delta:+.2f}% {arrow} {improved}（{len(values)} 个模型）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
