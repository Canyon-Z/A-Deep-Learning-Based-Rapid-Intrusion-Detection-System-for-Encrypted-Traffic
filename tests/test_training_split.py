"""训练切分与数据管线测试（缺陷 D-08，以及 `load_day` 的元信息对齐）。

核心不变量：**同一个时间块内的流不允许被拆到训练集和测试集两侧。**

旧实现 ``train_test_split(stratify=y, random_state=42)`` 会这么做，于是同一条 pcap
里相隔几十毫秒的流一半进训练、一半进测试，测试集上认出来的是孪生样本 ——
``logs/flow_multiclass_metrics.json`` 的 99.94% 就是这么来的。

这组测试用合成时间戳验证 ``block_split`` 的块内一致性，不需要真实 CSV。
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.train_multiclass import block_split  # noqa: E402


def _synthetic(n_per_class=3000, n_classes=3, block_seconds=60.0, seed=0):
    """构造「3 个类 × 连续时间戳」的样本，模拟同一次攻击的密集流。"""
    rng = np.random.RandomState(seed)
    ys, tss = [], []
    for cls in range(n_classes):
        # 每类占一小时，类内时间均匀分布
        base = pd.Timestamp('2018-03-02 01:00:00') + pd.Timedelta(hours=cls)
        offsets = np.sort(rng.uniform(0, 3600, n_per_class))
        tss.append(pd.Series(base + pd.to_timedelta(offsets, unit='s')))
        ys.append(np.full(n_per_class, cls, dtype=np.int64))
    y = np.concatenate(ys)
    meta = pd.DataFrame({'timestamp': pd.concat(tss, ignore_index=True),
                         'day': 'synthetic'})
    return y, meta


class TestBlockSplit(unittest.TestCase):
    def setUp(self):
        self.y, self.meta = _synthetic()
        self.assign = block_split(self.y, self.meta, test_size=0.2, val_size=0.15,
                                  block_seconds=60.0, seed=42)

    def test_every_sample_assigned(self):
        self.assertEqual(len(self.assign), len(self.y))
        self.assertEqual(set(self.assign.tolist()), {'train', 'val', 'test'})

    def test_no_block_straddles_split_boundary(self):
        """同 (类, 时间块) 的样本必须落在同一侧 —— 这是无泄漏的定义。"""
        ts = pd.to_datetime(self.meta['timestamp'])
        for cls in np.unique(self.y):
            idx = np.where(self.y == cls)[0]
            secs = ts.iloc[idx].astype('int64').to_numpy() / 1e9
            blocks = np.floor((secs - secs.min()) / 60.0).astype(int)
            for b in np.unique(blocks):
                sides = set(self.assign[idx[blocks == b]].tolist())
                self.assertEqual(
                    len(sides), 1,
                    f'类 {cls} 的时间块 {b} 被拆到了 {sides}，存在训练/测试泄漏')

    def test_each_class_present_on_all_sides(self):
        for cls in np.unique(self.y):
            for side in ('train', 'val', 'test'):
                n = int(((self.assign == side) & (self.y == cls)).sum())
                self.assertGreater(n, 0, f'类 {cls} 在 {side} 侧为空')

    def test_split_roughly_matches_requested_ratio(self):
        test_ratio = float((self.assign == 'test').mean())
        self.assertAlmostEqual(test_ratio, 0.2, delta=0.08)

    def test_deterministic_given_seed(self):
        again = block_split(self.y, self.meta, test_size=0.2, val_size=0.15,
                            block_seconds=60.0, seed=42)
        self.assertTrue((again == self.assign).all())

    def test_missing_timestamp_falls_back_to_row_blocks(self):
        """没有时间戳时退回「每 N 行一块」，仍不允许相邻行跨侧。"""
        meta = pd.DataFrame({'timestamp': pd.Series([pd.NaT] * len(self.y)),
                             'day': 'synthetic'})
        assign = block_split(self.y, meta, block_seconds=60.0, block_rows=500, seed=7)
        self.assertEqual(set(assign.tolist()), {'train', 'val', 'test'})
        # 每个块（500 行）内部必须同侧
        for start in range(0, len(assign), 500):
            chunk = assign[start:start + 500]
            self.assertEqual(len(set(chunk.tolist())), 1,
                             f'行块 [{start},{start + 500}) 被拆到了多侧')


class TestLoadDayMeta(unittest.TestCase):
    """`load_day(with_meta=True)` 必须返回与 DataFrame 行对齐的元信息。"""

    def test_meta_alignment(self):
        import csv as _csv
        import tempfile
        import glob  # noqa: F401
        from tools.train_flow_model import load_day

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, 'Thursday-01-03-2018_test.csv')
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = _csv.writer(fh)
            w.writerow(['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Label'])
            # 15/02 这种 dd/mm 必须按 day-first 解析，否则直接变 NaT
            w.writerow([80, 6, '15/02/2018 01:00:00', 100, 'BENIGN'])
            w.writerow([80, 6, '15/02/2018 01:00:30', 200, 'BENIGN'])
            w.writerow([80, 6, '15/02/2018 01:01:00', 300, 'BOT'])

        df, lab, meta = load_day(tmpdir, '01-03-2018', with_meta=True)
        self.assertEqual(len(df), len(meta))
        self.assertEqual(len(meta), 3)
        self.assertTrue(meta['timestamp'].notna().all(),
                        '时间戳出现 NaT，说明没有按 dd/mm/yyyy 解析')
        self.assertEqual(meta['timestamp'].iloc[0].month, 2)
        self.assertEqual(meta['timestamp'].iloc[0].day, 15)
        # 时间列必须已从特征里剔除，否则模型会学到「哪天」
        self.assertNotIn('Timestamp', df.columns)


if __name__ == '__main__':
    unittest.main()
