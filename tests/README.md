# 测试

```powershell
python -m unittest discover -s tests -t . -v
```

只依赖标准库 `unittest` 与项目既有的运行时依赖（numpy / torch / fastapi / httpx），
不需要额外装 pytest。CI 见 `.github/workflows/tests.yml`。

## 各文件锁住什么

| 文件 | 对应缺陷 | 不变量 |
| --- | --- | --- |
| `test_decision_semantics.py` | D-06 / D-09 / D-10 / D-11 | 标签只能由 `is_malicious` 翻译而来；`predicted_label` 与 `is_malicious` 永不矛盾 |
| `test_api_contract.py` | 验收报告参数校验缺口、M7 | `confidence_*` 必须在 [0,1] 且 low ≤ high；不存在的网卡不返回 `started`；锁定返回 429 + `Retry-After` |
| `test_cross_domain_caveats.py` | M4 / D-08 / D-18 / D-19 | 少数类过少、切分泄漏、比率口径越界、全零结果都必须被标注为不可引用 |
| `test_training_split.py` | D-08 | 同一时间块内的流不跨 train/test；`dd/mm/yyyy` 时间戳必须按 day-first 解析 |

## 为什么用 unittest

仓库原本没有任何测试，也没有声明 pytest 依赖。用标准库可以保证「克隆下来就能跑」，
不会因为测试工具本身缺失而让 CI 变成摆设。
