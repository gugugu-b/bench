"""配置常量 - 测试用例、数据集、请求数比例、CSV 表头、正则模式。

固定并发点扫描模式:所有要测的并发值都在 IO 里显式写好,不再做自适应搜索。
"""

import math
import time

# ============================================================
# 版本号
# ============================================================
VERSION = "v1.0.0"

# 数据集模式: 支持 "random" / "prefix_repetition",可填多个,按顺序各跑一遍
DATASET_MODES = ["random", "prefix_repetition"]

# 请求数 : 并发数 的全局默认比例(1:2 / 1:3 就填 2 / 3)
# 单个用例可以用 num_prompts_ratio 覆盖它
NUM_PROMPTS_RATIO = 2

# 某并发点已突破阈值时,是否提前结束该组合剩余的并发点
# False = 跑满配置的所有并发点(推荐,数据更完整); True = 一旦超阈值就停止本组合
STOP_ON_BREACH = False

# ============================================================
# 测试用例(固定输入输出组合 + 各自独立的并发列表)
# ============================================================
# 字段说明:
#   input_len        必填, 输入长度
#   output_len       必填, 输出长度
#   concurrency     必填, 该组合要跑的并发点列表(可任意增删调整); 写单个 int 等价于 [int]
#   num_prompts_ratio 可选, 请求数:并发数 比例, 不填用全局 NUM_PROMPTS_RATIO
#   ttft_max        可选, TTFT 阈值(ms), 不填用 DEFAULT_TTFT_MAX
#   tpot_max        可选, TPOT 阈值(ms), 不填用 DEFAULT_TPOT_MAX
DEFAULT_TTFT_MAX = 3000
DEFAULT_TPOT_MAX = 100

IO = [
    {
        "input_len": 1024,
        "output_len": 1024,
        "concurrency": [1, 4, 8, 16, 32, 64, 128],
        "num_prompts_ratio": 3,   # 请求数 = 并发 × 3
        "ttft_max": 3000,
        "tpot_max": 100,
    },
    {
        "input_len": 8192,
        "output_len": 1024,
        "concurrency": [1, 4, 8, 16, 32],
        "num_prompts_ratio": 2,   # 请求数 = 并发 × 2
        "ttft_max": 3000,
        "tpot_max": 100,
    },
]


def resolve_case(case: dict) -> dict:
    """把用户写的用例字典补全成完整配置(缺字段用全局默认值)。"""
    conc = case.get("concurrency") or []
    if isinstance(conc, int):
        conc = [conc]
    return {
        "input_len": int(case["input_len"]),
        "output_len": int(case["output_len"]),
        "concurrency": [int(c) for c in conc],
        "num_prompts_ratio": float(case.get("num_prompts_ratio") or NUM_PROMPTS_RATIO),
        "ttft_max": float(case.get("ttft_max", DEFAULT_TTFT_MAX)),
        "tpot_max": float(case.get("tpot_max", DEFAULT_TPOT_MAX)),
    }


def compute_num_prompts(concurrency: int, ratio: float) -> int:
    """请求数 = ceil(并发 × 比例);比例 ≥ 1 时保证不少于并发数。"""
    n = int(math.ceil(concurrency * ratio))
    return max(concurrency, n) if ratio >= 1 else max(1, n)


# 脚本启动时间戳(模块加载时计算一次,全包共享)
SCRIPT_START_TIME = time.strftime("%H%M%S")
SCRIPT_START_DATE = time.strftime("%Y%m%d")

# TTFT/TPOT 标签:可选 "Mean TTFT" / "Median TTFT" / "P99 TTFT" 等
TTFT_LABEL = "Mean TTFT"
TPOT_LABEL = "Mean TPOT"

# vllm bench serve 固定参数
HOST = "0.0.0.0"
PORT = "30000"
BACKEND = "vllm"
SERVED_MODEL_NAME = "DeepSeek-V4-Flash-Channel-FP8-w8a8"
MODEL = "/data/model/DeepSeek-V4-Flash-Channel-FP8-w8a8"
IGNORE_EOS = "--ignore-eos"

# ============================================================
# 数据集参数
# ============================================================
DATASET_NAME = "random"                              # random 模式 --dataset-name
PREFIX_REPETITION_DATASET_NAME = "prefix_repetition"  # 前缀重复模式 --dataset-name
PREFIX_REPETITION_PC_RATIO = 0.9                     # prefix 在输入长度中的占比
PREFIX_REPETITION_NUM_PREFIXES = 1

# ============================================================
# 执行参数
# ============================================================
MAX_RETRIES = 2                    # 单次测试失败后的重试次数
BENCH_MAX_ERRORS = MAX_RETRIES + 1  # 子进程连续失败上限,超出则抛 BenchmarkError
ENABLE_DOUBLE_RUN = True            # 第一次预热,第二次作为正式结果

# 运行时参数
SUBPROCESS_TIMEOUT = 3600       # vllm bench serve 子进程超时(秒)
POST_TEST_SLEEP = 2             # 单次测试后等待(秒)
RETRY_SLEEP = 2                 # 失败重试间隔(秒)

# perf_log 相关
PERF_LOG_DIR = "./bench/perf_log"
PERF_MODEL_NAME = "DeepSeek-V4-Flash-Channel-FP8-w8a8"  # 与 SERVED_MODEL_NAME 一致

# ============================================================
# CSV 表头
# ============================================================
_METRIC_COLUMNS = [
    "successful_requests", "benchmark_duration",
    "total_input_tokens", "total_generated_tokens",
    "req_throughput", "output_token_throughput", "total_token_throughput",
    "mean_ttft", "median_ttft", "p99_ttft",
    "mean_tpot", "median_tpot", "p99_tpot",
    "mean_itl", "median_itl", "p99_itl",
]

VLLM_BENCH_HEADERS = [
    "dataset", "input_len", "output_len", "concurrency", "num_prompts",
] + _METRIC_COLUMNS

# 每个并发点一行,passed=1 表示该点同时满足 TTFT/TPOT 阈值
SWEEP_HEADERS = [
    "dataset", "input_len", "output_len", "concurrency", "num_prompts",
    "ttft", "tpot", "passed",
]

SUMMARY_HEADERS = [
    "dataset", "input_len", "output_len", "num_prompts_ratio",
    "ttft_threshold", "tpot_threshold",
    "best_concurrency", "ttft", "tpot",
    "points_total", "points_passed",
]

BEST_METRICS_HEADERS = [
    "dataset", "input_len", "output_len", "concurrency", "num_prompts",
] + _METRIC_COLUMNS

# 指标正则:每个指标一个独立命名组,内层再命名一个数值捕获组
METRIC_PATTERNS = {
    'successful_requests': r"[Ss]uccessful\s+[Rr]equests?:\s*(\d+)",
    'benchmark_duration': r"[Bb]enchmark\s+[Dd]uration\s*\(?s\)?:\s*(\d+(?:\.\d+)?)",
    'total_input_tokens': r"[Tt]otal\s+[Ii]nput\s+[Tt]okens?:\s*(\d+)",
    'total_generated_tokens': r"[Tt]otal\s+[Gg]enerated\s+[Tt]okens?:\s*(\d+)",
    'req_throughput': r"[Rr]equest\s+[Tt]hroughput\s*\(req/s\)?:\s*(\d+(?:\.\d+)?)",
    'output_token_throughput': r"[Oo]utput\s+[Tt]oken\s+[Tt]hroughput\s*\(tok/s\)?:\s*(\d+(?:\.\d+)?)",
    'total_token_throughput': r"[Tt]otal\s+[Tt]oken\s+[Tt]hroughput\s*\(tok/s\)?:\s*(\d+(?:\.\d+)?)",
    'mean_ttft': r"[Mm]ean\s+TTFT\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
    'median_ttft': r"[Mm]edian\s+TTFT\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
    'p99_ttft': r"P99\s+TTFT\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
    'mean_tpot': r"[Mm]ean\s+TPOT\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
    'median_tpot': r"[Mm]edian\s+TPOT\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
    'p99_tpot': r"P99\s+TPOT\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
    'mean_itl': r"[Mm]ean\s+ITL\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
    'median_itl': r"[Mm]edian\s+ITL\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
    'p99_itl': r"P99\s+ITL\s*\(ms\)?:\s*(\d+(?:\.\d+)?)",
}

INT_METRIC_KEYS = frozenset({'successful_requests', 'total_input_tokens', 'total_generated_tokens'})

# TTFT/TPOT 标签 -> metrics 字典 key
TTFT_KEY_MAP = {
    "Mean TTFT": "mean_ttft",
    "Median TTFT": "median_ttft",
    "P99 TTFT": "p99_ttft",
}
TPOT_KEY_MAP = {
    "Mean TPOT": "mean_tpot",
    "Median TPOT": "median_tpot",
    "P99 TPOT": "p99_tpot",
}
