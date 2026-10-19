"""跨域评估结果的风险标注测试（缺陷 M4 / D-08 / D-18 / D-19）。

`/api/cross-domain` 把六份实验产物原样摆给页面。其中 image_ics 的类别分布是
``Benign 45497 / Malware 4``，于是「全部判成 Benign」就有 99.99% 的准确率，
MLP 那条 0.10% / FPR 99.91% 也只是一个失效模型的读数。这些数字过去没有任何
上下文说明，读者会把它们和 Pure_BiLSTM 的 99.37% 当成同一量级的能力指标。

``_cross_domain_caveats`` 不改原始指标（日志是实验证据，改动等于篡改），
只补出解读所需的上下文。这组测试锁住它的四类判定。
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.backend.main as main  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(key):
    path = main._CROSS_DOMAIN_FILES[key]
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


class TestCaveatRules(unittest.TestCase):
    """用合成数据锁判定规则，不依赖日志文件是否存在。"""

    def test_minority_class_flagged(self):
        data = {'class_counts': {'Benign': 45497, 'Malware': 4},
                'models': {'MLP': {'accuracy': 0.1033, 'fpr': 99.9055}}}
        c = main._cross_domain_caveats('image_ics', data)
        self.assertEqual(c['severity'], 'high')
        self.assertFalse(c['metrics_reliable'])
        self.assertEqual(c['minority_count'], 4)
        self.assertGreater(c['imbalance_ratio'], 1000)
        self.assertTrue(any('少数类' in n for n in c['notes']))
        self.assertTrue(any('MLP' in n for n in c['notes']))

    def test_healthy_result_not_flagged(self):
        data = {'class_counts': {'Benign': 5000, 'Malware': 4000},
                'models': {'Pure_BiLSTM': {'accuracy': 99.37, 'fpr': 0.633}}}
        c = main._cross_domain_caveats('image_cse', data)
        self.assertTrue(c['metrics_reliable'])
        self.assertEqual(c['notes'], ['该组指标的口径与样本量正常，可直接引用。'])

    def test_leaky_split_flagged(self):
        """没写 split_strategy 或写成 random，都要标成泄漏。"""
        for payload in ({'test_accuracy': 99.94},
                        {'test_accuracy': 99.94, 'split_strategy': 'random'}):
            c = main._cross_domain_caveats('flow_multiclass', payload)
            self.assertEqual(c['severity'], 'high')
            self.assertFalse(c['metrics_reliable'])
            self.assertTrue(any('泄漏' in n for n in c['notes']))

    def test_block_split_not_flagged(self):
        c = main._cross_domain_caveats(
            'flow_multiclass', {'test_accuracy': 88.0, 'split_strategy': 'block'})
        self.assertTrue(c['metrics_reliable'])

    def test_percent_unit_inside_subobject_not_flagged(self):
        """`default_threshold.fpr=1.91` 是百分比（train_autoencoder.py 统一 *100），不是越界。

        D-18 当初把它当成「0–1 比例下大于 1」，属单位误读；真正的缺陷是产物里没写单位。
        """
        c = main._cross_domain_caveats(
            'flow_autoencoder',
            {'workpoints': [{'fpr_limit': 5.0, 'tpr': 40.0, 'fpr': 1.91, 'threshold': 0.08}],
             'default_threshold': {'tpr': 40.0, 'fpr': 1.91, 'precision': 2.24}})
        self.assertEqual(c['severity'], 'info')
        self.assertTrue(c['metrics_reliable'])
        self.assertFalse(any('超过 100%' in n for n in c['notes']), c['notes'])
        self.assertFalse(any('比例' in n for n in c['notes']), c['notes'])

    def test_mixed_units_noted_but_not_blocking(self):
        """比率字段没声明单位时只给口径说明，不降低可信度、也不占用 notes。"""
        c = main._cross_domain_caveats(
            'flow_autoencoder',
            {'fpr': 0.0034,
             'default_threshold': {'tpr': 40.0, 'fpr': 15.35, 'precision': 2.24}})
        self.assertTrue(c['metrics_reliable'])
        self.assertIn('百分比', c.get('rate_unit_note') or '')

    def test_declared_unit_silences_note(self):
        c = main._cross_domain_caveats(
            'flow_autoencoder',
            {'rate_unit': 'percent', 'fpr': 1.91, 'precision': 2.24})
        self.assertEqual(c['severity'], 'info')
        self.assertIsNone(c.get('rate_unit_note'))

    def test_percent_scale_over_100_flagged(self):
        c = main._cross_domain_caveats('image_cse', {'fpr': 120.0, 'accuracy': 80.0})
        self.assertEqual(c['severity'], 'high')
        self.assertTrue(any('超过 100%' in n for n in c['notes']), c['notes'])

    def test_all_zero_adabn_flagged(self):
        c = main._cross_domain_caveats(
            'flow_adanb',
            {'baseline': {'accuracy': 72.0},
             'adanb': [{'accuracy': 0, 'detection_rate': 0, 'precision': 0}]})
        self.assertEqual(c['severity'], 'high')
        self.assertTrue(any('adaBN' in n for n in c['notes']))


class TestCaveatsWiredIntoResponse(unittest.TestCase):
    """端点必须真的把 caveats 带出去，否则前端拿不到。"""

    def test_route_declares_caveats(self):
        with open(os.path.join(_REPO, 'web', 'backend', 'main.py'), encoding='utf-8') as fh:
            src = fh.read()
        self.assertIn('"caveats"', src)
        for key in ('image_ics', 'image_cse', 'flow_multiclass', 'flow_adanb'):
            self.assertIn(f'"{key}": _cross_domain_caveats', src)


class TestRealArtifacts(unittest.TestCase):
    """对仓库里真实的实验产物跑一遍，确认当前结论与报告一致。"""

    def test_image_ics_is_unreliable(self):
        data = _load('image_ics')
        if data is None:
            self.skipTest('logs/external_eval_test_ics.json 不存在')
        c = main._cross_domain_caveats('image_ics', data)
        self.assertEqual(c['minority_count'], 4)
        self.assertFalse(c['metrics_reliable'])

    def test_flow_multiclass_leak_flag_follows_split_strategy(self):
        """泄漏标注要跟着产物的切分口径走，两个方向都得钉住。

        2026-10-19 之前 logs/flow_multiclass_metrics.json 没有 split_strategy 字段
        （按行随机切分，缺陷 D-08），本用例当时断言「必须标成不可信」。重训改成
        类内时间块切分后产物写成 ``split_strategy='block'`` / ``leakage_controlled=True``，
        同一个断言就变成「新口径也必须当作泄漏」——反了。所以这里改成双向断言：
        真实产物（block）→ 可信；伪造一份没有切分字段的旧产物 → 仍必须标不可信。
        """
        data = _load('flow_multiclass')
        if data is None:
            self.skipTest('logs/flow_multiclass_metrics.json 不存在')

        c = main._cross_domain_caveats('flow_multiclass', data)
        self.assertEqual(data.get('split_strategy'), 'block')
        self.assertTrue(data.get('leakage_controlled'))
        self.assertTrue(
            c['metrics_reliable'],
            f"block 切分产物的可信标记仍为 False：{c['notes']}")

        legacy = dict(data)
        legacy.pop('split_strategy', None)
        legacy.pop('leakage_controlled', None)
        c_legacy = main._cross_domain_caveats('flow_multiclass', legacy)
        self.assertFalse(c_legacy['metrics_reliable'])
        self.assertEqual(c_legacy['severity'], 'high')


if __name__ == '__main__':
    unittest.main()
