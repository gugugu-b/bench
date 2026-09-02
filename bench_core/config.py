"""配置常量 - 测试用例、数据集、请求数比例、CSV 表头、正则模式。

固定并发点扫描模式:所有要测的并发值都在 IO 里显式写好,不再做自适应搜索。
"""

import math
import time

# ============================================================
# 版本号
# ============================================================
VERSION = "v1.2.0"

# 数据集模式: 支持 "random" / "prefix_repetition",可填多个,按顺序各跑一遍
# 用例可用 datasets 字段指定自己要跑的数据集列表,不填的用例用这里的全局默认
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
#   pc_ratio        可选, prefix_repetition 模式下 prefix 在输入长度中的占比, 不填用 PREFIX_REPETITION_PC_RATIO
#   num_prefixes    可选, prefix_repetition 模式的前缀数, 不填用 PREFIX_REPETITION_NUM_PREFIXES
#   datasets        可选, 该用例要跑的数据集列表, 不填用全局 DATASET_MODES
#   warmup_rounds   可选, 预热轮数(int 或 {dataset: 轮数}), 不填用全局 WARMUP_ROUNDS
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
        "pc_ratio": 0.1,          # prefix 占输入长度的 10%
        "datasets": ["prefix_repetition"],
    },
    {
        "input_len": 1024,
        "output_len": 1024,
        "concurrency": [1, 4, 8, 16, 32, 64, 128],
        "num_prompts_ratio": 3,   # 请求数 = 并发 × 3
        "ttft_max": 3000,
        "tpot_max": 100,
        "pc_ratio": 0.9,          # prefix 占输入长度的 90%
        "datasets": ["prefix_repetition"],
    },
    {
        "input_len": 8192,
        "output_len": 1024,
        "concurrency": [1, 4, 8, 16, 32],
        "num_prompts_ratio": 2,   # 请求数 = 并发 × 2
        "ttft_max": 3000,
        "tpot_max": 100,
        "datasets": ["random", "prefix_repetition"],
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
        "pc_ratio": float(case.get("pc_ratio", PREFIX_REPETITION_PC_RATIO)),
        "num_prefixes": int(case.get("num_prefixes", PREFIX_REPETITION_NUM_PREFIXES)),
        "datasets": list(case.get("datasets") or DATASET_MODES),
        "warmup_rounds": case.get("warmup_rounds"),
    }


def compute_num_prompts(concurrency: int, ratio: float) -> int:
    """请求数 = ceil(并发 × 比例);比例 ≥ 1 时保证不少于并发数。"""
    n = int(math.ceil(concurrency * ratio))
    return max(concurrency, n) if ratio >= 1 else max(1, n)


def prefix_context_tag(dataset: str, pc_ratio: float, num_prefixes: int) -> str:
    """prefix_repetition 数据集的上下文目录后缀。

    输入输出相同但前缀参数不同的用例,靠该后缀区分 context 目录与汇总表行;
    random 等其他数据集返回空串。
    """
    if dataset != PREFIX_REPETITION_DATASET_NAME:
        return ""
    return f"_pc{pc_ratio:g}_np{num_prefixes}"


def dataset_prefix_fields(dataset: str, pc_ratio: float, num_prefixes: int):
    """前缀参数在 CSV 标识列中的取值。

    仅 prefix_repetition 数据集有意义,返回原值;random 等其他数据集返回空串,
    避免把「未参与测试的回退默认值」误记为有效配置。
    """
    if dataset != PREFIX_REPETITION_DATASET_NAME:
        return "", ""
    return pc_ratio, num_prefixes


def resolve_warmup_rounds(case_rounds, dataset: str) -> int:
    """取某数据集的预热轮数(仅 ENABLE_DOUBLE_RUN=True 时生效)。

    用例级 warmup_rounds 与全局 WARMUP_ROUNDS 都可为 int(统一轮数)
    或 {dataset: 轮数};用例级优先,其 dict 未覆盖该数据集时回退全局,
    两级都未覆盖时为 1 轮。填 0 表示该场景不预热。
    """
    for cfg in (case_rounds, WARMUP_ROUNDS):
        if cfg is None:
            continue
        if isinstance(cfg, dict):
            if dataset in cfg:
                return int(cfg[dataset])
        else:
            return int(cfg)
    return 1


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
ENABLE_DOUBLE_RUN = True            # 开启预热:正式测试前先用相同命令预热若干轮

# 预热轮数(仅 ENABLE_DOUBLE_RUN=True 时生效)
# 可为 int(所有数据集统一轮数)或 dict(按数据集指定,未列出的数据集回退 1 轮),如:
#   WARMUP_ROUNDS = 1
#   WARMUP_ROUNDS = {"random": 1, "prefix_repetition": 4}
# 用例级可用 warmup_rounds 字段覆盖,写法相同;填 0 表示该场景不预热
WARMUP_ROUNDS = {"random": 1, "prefix_repetition": 4}

# 运行时参数
SUBPROCESS_TIMEOUT = 3600       # vllm bench serve 子进程超时(秒)
POST_TEST_SLEEP = 2             # 单次测试后等待(秒)

# 从被测服务的 Prometheus /metrics 抓取指标(vLLM API server 默认在同一 host:port 暴露),
# 用正式测试前后快照的差值计算 prefix cache 命中率与投机采样接受率;
# 抓取失败只告警一次,不影响测试,对应列留空
ENABLE_METRICS_SCRAPE = True    # 是否抓取 /metrics
METRICS_SCRAPE_PATH = "/metrics"  # 抓取路径
METRICS_SCRAPE_TIMEOUT = 5      # 抓取超时(秒)
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

# 每个成功并发点一行:关键性能指标 + 单并发归一化吞吐
# pc_ratio / num_prefixes 为该用例的前缀重复参数(random 数据集下不参与命令,仅作标识)
# output_throughput_per_concurrency = output_token_throughput / concurrency
# decode_throughput_per_concurrency = 1000 / mean_tpot,即单条请求流的 decode 速率(tok/s)
# prefix_cache_hit_rate / spec_decode_accept_rate 来自 /metrics 正式测试前后快照差值(百分数,不可用为空)
POINT_METRICS_HEADERS = [
    "dataset", "input_len", "output_len", "concurrency", "pc_ratio", "num_prefixes",
    "mean_ttft", "mean_tpot",
    "output_token_throughput", "total_token_throughput", "benchmark_duration",
    "output_throughput_per_concurrency", "decode_throughput_per_concurrency",
    "prefix_cache_hit_rate", "spec_decode_accept_rate",
]

SUMMARY_HEADERS = [
    "dataset", "input_len", "output_len", "pc_ratio", "num_prefixes", "num_prompts_ratio",
    "ttft_threshold", "tpot_threshold",
    "best_concurrency", "ttft", "tpot",
    "points_total", "points_passed",
]

BEST_METRICS_HEADERS = [
    "dataset", "input_len", "output_len", "pc_ratio", "num_prefixes",
    "concurrency", "num_prompts",
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
