"""vllm_benchmark 固定并发点扫描测试 - 入口

按 config.IO 里配置好的输入输出组合与并发列表逐点测试。
实际逻辑在 bench_core 包内,本文件仅作为薄入口。
"""

import logging

from bench_core import run_test_cases

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    run_test_cases()
