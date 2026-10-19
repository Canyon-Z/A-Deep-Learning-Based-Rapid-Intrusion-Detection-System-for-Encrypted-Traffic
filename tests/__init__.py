"""项目测试包。

运行方式（在仓库根目录）：

    python -m unittest discover -s tests -t . -v

只依赖标准库 unittest + 项目已声明的运行时依赖（numpy/torch/fastapi/httpx），
不需要额外安装 pytest。CI 见 `.github/workflows/tests.yml`。
"""
