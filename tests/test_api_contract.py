"""HTTP 接口契约测试（验收报告「参数校验缺口」+ M7 登录锁定）。

锁住三件曾经出过问题的事：

1. ``/analyze-flow`` 的 ``confidence_high / confidence_low`` 传 2、传 -1，
   或者 low > high，过去一律 200 并生效，档位判定整段错乱；
2. ``/api/live/start`` 传一个不存在的网卡，过去返回 ``{"status": "started"}``，
   真实错误只藏在 ``/api/live/debug`` 的 last_error 里；
3. 登录失败达到上限时，过去返回 200 + 一段错误文案，脚本与监控无法识别限流。

这些用例用 ``fastapi.testclient``，并把 ``require_session`` 依赖替换成固定用户，
避免为测参数校验而先搭一套登录态。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import web.backend.main as main  # noqa: E402


def _client():
    app = main.app
    app.dependency_overrides[main.require_session] = lambda: 'test-user'
    return TestClient(app)


class TestAnalyzeFlowValidation(unittest.TestCase):
    """`/analyze-flow` 的门槛参数必须在 [0,1] 且 low <= high。"""

    def setUp(self):
        self.client = _client()
        # 上传一个最小的合法 CSV：只要签名能过就够，校验发生在读文件之前
        self.csv = b'Label,Dst Port\nBenign,80\n'

    def _post(self, **form):
        return self.client.post('/analyze-flow', data=form,
                                files={'file': ('t.csv', self.csv, 'text/csv')})

    def test_high_above_one_rejected(self):
        r = self._post(confidence_high=2.0)
        self.assertEqual(r.status_code, 400, r.text)

    def test_negative_low_rejected(self):
        r = self._post(confidence_low=-0.1)
        self.assertEqual(r.status_code, 400, r.text)

    def test_inverted_thresholds_rejected(self):
        r = self._post(confidence_high=0.2, confidence_low=0.9)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn('confidence_low', r.text)

    def test_valid_thresholds_not_rejected_by_validation(self):
        """合法参数不应被门槛校验拦下（缺列是另一类错误，与信任边界无关）。"""
        r = self._post(confidence_high=0.95, confidence_low=0.6)
        body = r.text
        self.assertNotIn('confidence_high / confidence_low must be within', body)
        self.assertNotIn('confidence_low must be <= confidence_high', body)

    def test_threshold_check_precedes_missing_file(self):
        """不带文件时，非法门槛仍必须报门槛错误（400），而不是被 422「缺文件」盖住。

        回归价值：file 曾声明为 `File(...)`，FastAPI 在依赖解析阶段就回 422，
        函数体里的门槛校验永远跑不到，用户看到的是「缺文件」而非真正的原因。
        """
        r = self.client.post('/analyze-flow', data={'confidence_high': 2.0})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn('must be within', r.text)

    def test_missing_file_reported_when_params_valid(self):
        r = self.client.post('/analyze-flow', data={'confidence_high': 0.9})
        self.assertEqual(r.status_code, 422, r.text)
        self.assertIn('file is required', r.text)


class TestLiveStartValidation(unittest.TestCase):
    """不存在的网卡必须当场报错，不能返回 started。"""

    def setUp(self):
        self.client = _client()
        try:
            rows = main._collect_interface_rows()
        except Exception:
            rows = []
        if not rows:
            self.skipTest('本机列不出网卡，无从校验 iface 存在性')

    def test_unknown_interface_rejected(self):
        r = self.client.post('/api/live/start', data={'iface': '__no_such_iface__'})
        self.assertEqual(r.status_code, 400, r.text)
        body = r.json()
        self.assertEqual(body.get('status'), 'error')
        self.assertIn('__no_such_iface__', body.get('message', ''))


class TestLoginLockout(unittest.TestCase):
    """连续失败到上限后，必须用 429 + Retry-After 表达限流（缺陷 M7）。"""

    def setUp(self):
        self.client = _client()
        main._login_failures.clear()

    def tearDown(self):
        main._login_failures.clear()

    def test_lockout_returns_429_with_retry_after(self):
        limit = main.LOGIN_MAX_FAILURES
        last = None
        for _ in range(limit + 1):
            last = self.client.post('/login', data={'username': '__nobody__',
                                                    'password': '__wrong__'})
        self.assertEqual(last.status_code, 429, last.text[:200])
        self.assertIsNotNone(last.headers.get('Retry-After'))
        self.assertGreater(int(last.headers['Retry-After']), 0)


class TestInferenceQueue(unittest.TestCase):
    """推理并发闸门（缺陷 M6）。

    原先 4 路并发把单次 5.8s 拖到 30.7s，且没有任何排队/拒绝语义。
    这里锁住两条最小事实：推理入口在闸门名单里；闸门被占满且排队超时后回 429
    并带 Retry-After，而不是无限等待或静默串行。
    """

    def setUp(self):
        self.client = _client()

    def tearDown(self):
        # 恢复信号量，避免污染后续用例
        main._inference_semaphore = __import__('asyncio').Semaphore(
            main.MAX_CONCURRENT_INFERENCE)
        main.INFERENCE_QUEUE_TIMEOUT = 120.0

    def test_guarded_paths_declared(self):
        self.assertIn('/analyze', main.INFERENCE_GUARDED_PATHS)
        self.assertIn('/analyze-flow', main.INFERENCE_GUARDED_PATHS)

    def test_queue_timeout_returns_429(self):
        # 手工把信号量占满，再把排队超时压到极小，模拟「队列已满」
        main._inference_semaphore = __import__('asyncio').Semaphore(1)
        main._inference_semaphore._value = 0
        main.INFERENCE_QUEUE_TIMEOUT = 0.01
        r = self.client.post('/analyze-flow', data={'confidence_high': 0.9},
                             files={'file': ('t.csv', b'Label,Dst Port\nBenign,80\n',
                                             'text/csv')})
        self.assertEqual(r.status_code, 429, r.text[:200])
        self.assertIn('Retry-After', r.headers)
        self.assertEqual(r.json().get('max_concurrent_inference'),
                         main.MAX_CONCURRENT_INFERENCE)

    def test_unguarded_path_not_blocked(self):
        """闸门只作用于推理入口，页面与统计接口不该被排队拖住。"""
        main._inference_semaphore = __import__('asyncio').Semaphore(1)
        main._inference_semaphore._value = 0
        main.INFERENCE_QUEUE_TIMEOUT = 0.01
        r = self.client.get('/login')
        self.assertNotEqual(r.status_code, 429)


class TestRecordsPersistence(unittest.TestCase):
    """检测记录的服务端持久化（缺口 L2）。

    原先 /records 只有页面渲染，数据全在 localStorage：换机器就丢。
    这里锁住最小契约：空态返回空结构、写入后可读回、超限被拒、可删除。
    """

    def setUp(self):
        self.client = _client()
        self.user = 'test-user'
        self.path = main._records_path(self.user)
        self._backup = None
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                self._backup = f.read()

    def tearDown(self):
        try:
            if self._backup is None:
                if os.path.exists(self.path):
                    os.remove(self.path)
            else:
                with open(self.path, 'wb') as f:
                    f.write(self._backup)
        except Exception:
            pass

    def test_roundtrip_and_delete(self):
        self.client.delete('/api/records')
        empty = self.client.get('/api/records').json()
        self.assertEqual(empty['live'], [])
        self.assertEqual(empty['upload'], [])
        self.assertEqual(empty['storage'], 'server')

        payload = {'live': [{'data': {'src': '10.0.0.1'}}], 'upload': [{'data': {'name': 'a.pcap'}}]}
        saved = self.client.post('/api/records', json=payload)
        self.assertEqual(saved.status_code, 200, saved.text[:200])
        self.assertTrue(saved.json()['ok'])

        back = self.client.get('/api/records').json()
        self.assertEqual(len(back['live']), 1)
        self.assertEqual(len(back['upload']), 1)
        self.assertIsNotNone(back['updated_at'])

        self.client.delete('/api/records')
        self.assertEqual(self.client.get('/api/records').json()['live'], [])

    def test_oversized_payload_rejected(self):
        original = main.RECORDS_MAX_BYTES
        main.RECORDS_MAX_BYTES = 64
        try:
            r = self.client.post('/api/records', json={'live': [{'x': 'y' * 200}], 'upload': []})
            self.assertEqual(r.status_code, 413, r.text[:200])
        finally:
            main.RECORDS_MAX_BYTES = original

    def test_non_object_body_rejected(self):
        r = self.client.post('/api/records', json=[1, 2, 3])
        self.assertEqual(r.status_code, 400, r.text[:200])


if __name__ == '__main__':
    unittest.main()
