"""判决口径一致性测试（缺陷 D-06 / D-09 / D-10 / D-11）。

背景：验收报告 B2 指出，同一条流在页面上会同时出现「二分类判 Malware 99.95%」与
「六分类威胁卡 Benign 100%」。根因之一是**标签有两个来源**：

    predicted_label = class_names[argmax]          # 旧：raw argmax
    is_malicious    = argmax == malware and conf >= threshold

当 ``0.5 < malware_conf < threshold`` 时，标签写 Malware、布尔判定却是 False，
前端的弹窗与统计表各读一个字段，自然打架。

修复后只有一个真源：先按阈值算 ``is_malicious``，再由
``_decision_label(is_malicious)`` 翻译成标签。这组测试锁住这条不变量。
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.backend.main as main  # noqa: E402


class TestDecisionLabel(unittest.TestCase):
    def test_label_maps_from_boolean_only(self):
        self.assertEqual(main._decision_label(True), main.DECISION_POSITIVE_LABEL)
        self.assertEqual(main._decision_label(False), main.DECISION_NEGATIVE_LABEL)

    def test_labels_are_the_two_known_strings(self):
        # 前端 main.js / modal.js 只认这两个字符串；改名前必须先改前端。
        self.assertEqual(main.DECISION_POSITIVE_LABEL, 'Malicious')
        self.assertEqual(main.DECISION_NEGATIVE_LABEL, 'Benign')


class TestSingleSessionDecision(unittest.TestCase):
    """`_decide_single_session(argmax_idx, probs, malware_idx, threshold)`。"""

    def _probs(self, benign, malware):
        p = np.zeros(2, dtype=np.float32)
        p[0], p[1] = benign, malware
        return p

    def test_high_confidence_attack(self):
        is_mal, argmax_label, decision = main._decide_single_session(
            1, self._probs(0.02, 0.98), 1, 0.65)
        self.assertTrue(is_mal)
        self.assertEqual(argmax_label, 'Malware')
        self.assertEqual(decision, main.DECISION_POSITIVE_LABEL)

    def test_confident_benign(self):
        is_mal, argmax_label, decision = main._decide_single_session(
            0, self._probs(0.97, 0.03), 1, 0.65)
        self.assertFalse(is_mal)
        self.assertEqual(argmax_label, 'Benign')
        self.assertEqual(decision, main.DECISION_NEGATIVE_LABEL)

    def test_gray_zone_never_reports_attack_label(self):
        """argmax 是恶意、但没到阈值 —— 这是 B2 矛盾的原始触发条件。

        旧实现会返回 ``('Malware', False)``；现在必须返回 ``('Benign', False)``，
        即：标签与布尔判定不可能互相矛盾。
        """
        is_mal, argmax_label, decision = main._decide_single_session(
            1, self._probs(0.35, 0.65), 1, 0.90)
        self.assertFalse(is_mal)
        self.assertEqual(argmax_label, 'Malware')   # argmax 事实照实返回
        self.assertEqual(decision, main.DECISION_NEGATIVE_LABEL)  # 但结论是 Benign
        self.assertNotEqual(decision, main.DECISION_POSITIVE_LABEL)

    def test_decision_always_consistent_with_boolean(self):
        rng = np.random.RandomState(0)
        for _ in range(200):
            probs = rng.dirichlet([1.0, 1.0]).astype(np.float32)
            argmax = int(probs.argmax())
            threshold = float(rng.uniform(0.5, 0.99))
            is_mal, _, decision = main._decide_single_session(argmax, probs, 1, threshold)
            self.assertEqual(decision, main._decision_label(is_mal))


class TestThresholdConfig(unittest.TestCase):
    """配置层也必须有校验：模块级别的 low > high 会自动交换。"""

    def test_module_thresholds_ordered(self):
        self.assertLessEqual(main.FLOW_CONFIDENCE_LOW, main.FLOW_CONFIDENCE_HIGH)

    def test_classify_matrix_swaps_inverted_thresholds(self):
        """`_classify_flow_matrix` 内部对传反的阈值做交换，而不是照单全收。"""
        if main.flow_multiclass is None:
            self.skipTest('flow_multiclass checkpoint not available')
        n_feat = len(main.flow_multiclass_meta['feature_names'])
        X = np.random.RandomState(1).rand(8, n_feat).astype(np.float32)
        summary, _detail = main._classify_flow_matrix(X, confidence_high=0.2, confidence_low=0.9)
        # 交换后 high=0.9 / low=0.2；六个类别都在返回值里（不因阈值非法而崩）
        self.assertTrue(summary)
        for name in main.flow_multiclass_meta['class_names']:
            self.assertIn(name, summary)


if __name__ == '__main__':
    unittest.main()
