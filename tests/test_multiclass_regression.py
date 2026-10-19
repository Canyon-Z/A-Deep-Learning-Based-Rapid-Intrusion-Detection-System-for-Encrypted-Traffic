"""B1 回归：演示集上的六分类准确率、未知攻击占比与各类 recall。

验收报告的阻塞项 B1 是这样的：`data/cse_csv/demo_mixed_6class.csv` 上整体准确率
只有 65.79%，44.3% 的样本被判成「未知攻击」，DoS-Slowloris 的 recall 只有 16.0%
（270/399 被判成 Benign）。根因是 `checkpoints/flow_env_calib.npz` 在启动时覆盖了
权重自带的标准화参数（缺陷 D-01），输入与 `flow_class_centers.npz` 处在两个不同的
坐标系里，距离判据于是把大片正常流量推成「未知」。

**2026-10-19 重铸（切分口径改为 block 之后）**

上面那个 65.79% 修好以后本用例锁的下限是「整体 ≥95%、每类 recall ≥90%」，
但那是**按行随机切分**时期定的门槛，而这份演示集本身就来自训练用的那三个 CSV：

* 3600 行里有 3191 行（88.64%）能在训练 CSV 里逐字节找到，五个恶意类的 400 行
  全部命中，剩下 409 行落在 Benign 上（NaN/inf 清洗后未入库）；
* 于是「按行随机切分」时这些行同时在训练集与测试集里，98.69% 量的是记忆而非泛化。

换成类内时间块切分（缺陷 D-08 的修法）后，实测（见 `logs/report_block_split.md`、
`logs/_diag5.txt`）：

* 已被训练过的块：1839/1861 = 98.8%（旧基线 98.69% 就落在这里，两边正好对上）；
* 从未见过的块：1053/1230 = 85.6%，整体 3600 行 94.47%；
* 唯一塌掉的是 DoS-Slowloris：全量 10990 条里 98.5% 判对，误判的 160 条**集中在
  11:00:00 这一个 60s 块**（Slowloris 起手那一分钟，502 条，块恰好被分到验证侧）；
  演示集抽到了这个块里的 279 行，recall 因此从 100% 掉到 60.25%。

所以这里把门槛改成「实测值 + 明确留白」，既不放任静默退化，也不再假装 90% recall
是无泄漏时的常态：

* 整体准确率 ≥ 93%（实测 94.47%；旧随机切分口径下是 98.69%）
* 未知攻击占比 ≤ 5%（实测 1.53%）
* DDoS-LOIC-UDP / Bot / DDoS-HOIC / DoS-GoldenEye ≥ 90%，Benign ≥ 90%，
  DoS-Slowloris ≥ 55%（实测 60.25%，全量口径 98.5%）
* OOD 阈值必须等于 `checkpoints/flow_class_centers.json` 里的 p99.9，不再手写常数
  —— 重训会把它从 31.53 挪到 37.77，写死就等于把自己钉在旧权重上。

`demo_mixed_6class.csv` 里混着 3 行「重复表头」（数值列取值等于列名本身），
那是数据生成时留下的脏行。用例会先把它们剔掉，同时断言这些脏行不超过 5 行 ——
如果哪天变多，说明有人重写了这个演示集而没有清洗。
"""

import json
import os
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV = os.path.join(_REPO, 'data', 'cse_csv', 'demo_mixed_6class.csv')
_CENTERS_JSON = os.path.join(_REPO, 'checkpoints', 'flow_class_centers.json')

# CSV 里的原始标签名 → 模型 class_names
LABEL_MAP = {
    'Benign': 'Benign',
    'Bot': 'Bot',
    'DDOS attack-HOIC': 'DDoS-HOIC',
    'DOS attack-HOIC': 'DDoS-HOIC',
    'DoS attacks-GoldenEye': 'DoS-GoldenEye',
    'DoS attacks-Slowloris': 'DoS-Slowloris',
    'DDOS attack-LOIC-UDP': 'DDoS-LOIC-UDP',
}

# 每类的 recall 下限：见模块 docstring 的实测依据。
# DoS-Slowloris 单独放宽，是因为它在演示集里的 279/400 行来自「起手一分钟」那个
# 没训练过的块 —— 门槛写成 90% 等于要求模型把训练时从未见过的流量形态也认全，
# 那不是回归测试，是许愿。
RECALL_FLOOR = {
    'Benign': 90.0,
    'Bot': 90.0,
    'DDoS-HOIC': 90.0,
    'DoS-GoldenEye': 90.0,
    'DoS-Slowloris': 55.0,
    'DDoS-LOIC-UDP': 90.0,
}

OVERALL_FLOOR = 93.0
UNKNOWN_CEILING = 5.0


def _load_demo():
    """返回 (X, y, 脏行数)；模型或数据缺失时返回 None。"""
    try:
        import pandas as pd

        import web.backend.main as main
    except Exception:  # pragma: no cover - 依赖缺失时跳过
        return None
    if main.flow_multiclass is None or main.flow_multiclass_meta is None:
        return None
    if not os.path.exists(_CSV):
        return None

    feats = list(main.flow_multiclass_meta['feature_names'])
    df = pd.read_csv(_CSV, low_memory=False)
    num = df[feats].apply(lambda s: pd.to_numeric(s, errors='coerce'))
    noise = (num.isna() & df[feats].notna()).any(axis=1)

    X = num.loc[~noise].to_numpy(dtype=np.float32)
    y = df.loc[~noise, 'Label'].map(
        lambda s: LABEL_MAP.get(str(s).strip(), str(s).strip())).to_numpy()
    return X, y, int(noise.sum())


def _artifact_p999():
    """类中心产物里的 p99.9 距离阈值；缺产物返回 None。"""
    if not os.path.exists(_CENTERS_JSON):
        return None
    with open(_CENTERS_JSON, encoding='utf-8') as fh:
        data = json.load(fh)
    return float(data['distance']['percentiles']['99.9'])


class TestMulticlassRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = _load_demo()

    def _require(self):
        if self.loaded is None:
            self.skipTest('演示集或六分类权重不可用')
        X, y, noise = self.loaded
        import web.backend.main as main
        _, detail = main._classify_flow_matrix(X)
        pred = np.array([detail['class_names'][i] for i in detail['pred']])
        return y, pred, detail, noise

    def test_demo_csv_has_no_header_noise(self):
        _, _, _, noise = self._require()
        self.assertLessEqual(
            noise, 5, f'演示集里混入了 {noise} 行重复表头，数据需要清洗')

    def test_overall_accuracy(self):
        y, pred, _, _ = self._require()
        acc = float((pred == y).mean()) * 100.0
        self.assertGreaterEqual(
            acc, OVERALL_FLOOR,
            f'六分类整体准确率 {acc:.2f}% 低于 {OVERALL_FLOOR}%'
            '（block 切分实测 94.47%，随机切分时期的 98.69% 含泄漏）')

    def test_unknown_ratio(self):
        _, _, detail, _ = self._require()
        ratio = float(detail['is_unknown'].mean()) * 100.0
        self.assertLessEqual(
            ratio, UNKNOWN_CEILING,
            f'未知攻击占比 {ratio:.2f}% 超过 {UNKNOWN_CEILING}%（修复前是 44.3%）')

    def test_per_class_recall(self):
        y, pred, detail, _ = self._require()
        for name in detail['class_names']:
            mask = y == name
            if not mask.any():
                continue
            floor = RECALL_FLOOR.get(name, 90.0)
            recall = float((pred[mask] == name).mean()) * 100.0
            self.assertGreaterEqual(
                recall, floor, f'{name} 的 recall {recall:.2f}% 低于 {floor}%')

    def test_ood_gate_is_enabled_and_uses_p999(self):
        """OOD 必须真的启用，且阈值等于类中心产物的 p99.9（不是手写常数）。

        旧版本在这里写死 31.5347，重训一次中心就失败一次；阈值本来就该从产物读，
        断言的是「运行时取的就是产物里那个数」，而不是某个特定权重的历史值。
        """
        _, _, detail, _ = self._require()
        expected = _artifact_p999()
        if expected is None:
            self.skipTest('缺少 checkpoints/flow_class_centers.json')
        self.assertTrue(detail['ood_enabled'])
        self.assertIsNone(detail['ood_disabled_reason'])
        self.assertAlmostEqual(detail['ood_threshold'], expected, places=3)

        # 同一个用例里锁住「类中心与当前权重处在同一坐标系」：一旦两者不同空间，
        # space_matches_active=False 会把整条距离支路关掉，上面的阈值断言也就没意义了。
        import web.backend.main as main
        self.assertTrue(
            (main.flow_centers or {}).get('space_matches_active'),
            'flow_class_centers.npz 与 flow_multiclass.pth 的 normalization 不一致，'
            '需要重跑 tools/build_flow_centers.py')


if __name__ == '__main__':
    unittest.main()
