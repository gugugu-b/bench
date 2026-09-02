"""Benchmark 执行 - 调用 vllm bench serve 子进程,提取指标,写 CSV/perf_log。"""

import logging
import math
import os
import subprocess
import time
from typing import Tuple

from .config import (
    BACKEND,
    BENCH_MAX_ERRORS,
    DATASET_NAME,
    ENABLE_DOUBLE_RUN,
    ENABLE_METRICS_SCRAPE,
    HOST,
    IGNORE_EOS,
    MAX_RETRIES,
    METRICS_SCRAPE_PATH,
    METRICS_SCRAPE_TIMEOUT,
    MODEL,
    PERF_LOG_DIR,
    PERF_MODEL_NAME,
    PORT,
    POST_TEST_SLEEP,
    PREFIX_REPETITION_DATASET_NAME,
    RETRY_SLEEP,
    SERVED_MODEL_NAME,
    SUBPROCESS_TIMEOUT,
    SWEEP_HEADERS,
    TTFT_LABEL,
    TPOT_LABEL,
    VLLM_BENCH_HEADERS,
    compute_num_prompts,
    prefix_context_tag,
)
from .csv_io import write_to_csv
from .metrics import (
    _extract_all_metrics,
    compute_metrics_rates,
    scrape_prometheus_metrics,
    select_ttft,
    select_tpot,
)


class BenchmarkError(Exception):
    """benchmark 执行过程中的不可恢复错误。"""
    pass


# 子进程连续失败计数,达到 BENCH_MAX_ERRORS 抛 BenchmarkError
_bench_err_num = 0

# /metrics 抓取失败是否已告警过(只告警一次,避免每个测试点刷屏)
_metrics_scrape_warned = False


def _scrape_metrics_snapshot():
    """抓取一次 /metrics 快照;开关关闭或抓取失败返回 None(失败只告警一次)。"""
    global _metrics_scrape_warned
    if not ENABLE_METRICS_SCRAPE:
        return None
    try:
        return scrape_prometheus_metrics(HOST, PORT, METRICS_SCRAPE_PATH, METRICS_SCRAPE_TIMEOUT)
    except Exception as e:
        if not _metrics_scrape_warned:
            logging.warning(
                f"抓取 http://{HOST}:{PORT}{METRICS_SCRAPE_PATH} 失败: {e}; "
                "prefix cache 命中率与投机采样接受率将留空"
                "(服务未暴露 /metrics 时可用 ENABLE_METRICS_SCRAPE=False 关闭抓取)"
            )
            _metrics_scrape_warned = True
        return None


def reset_bench_error_counter():
    """每个测试用例开始时调用,重置错误计数。"""
    global _bench_err_num
    _bench_err_num = 0


def save_perf_log_entry(input_len: int, output_len: int, concurrency: int, num_prompts: int,
                        dataset: str, metrics: dict, raw_output: str, context_suffix: str = ""):
    """保存 perf_log 格式的日志条目(原始输出 + 提取的指标)。

    目录: perf_log/<模型名>_<dataset>{context_suffix}/  文件名: il{input_len}_ol{output_len}_np{np}_mc{concurrency}.log
    文件名格式固定,供外部导入使用;同名用例(如不同 pc_ratio)靠目录后缀区分。
    """
    log_dir = os.path.join(PERF_LOG_DIR, f"{PERF_MODEL_NAME}_{dataset}{context_suffix}")
    os.makedirs(log_dir, exist_ok=True)
    sub_log_file = os.path.join(
        log_dir, f"il{input_len}_ol{output_len}_np{num_prompts}_mc{concurrency}.log"
    )
    with open(sub_log_file, 'w', encoding='utf-8') as f:
        f.write(f"Input Length: {input_len}\n")
        f.write(f"Output Length: {output_len}\n")
        f.write(f"Concurrency: {concurrency}\n")
        f.write(f"Num Prompts: {num_prompts}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        f.write("Raw Benchmark Output:\n")
        f.write(raw_output)
        f.write("=" * 50 + "\n")
        f.write("Extracted Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def _build_bench_cmd(input_len: int, output_len: int, concurrency: int,
                     dataset: str, num_prompts: int, pc_ratio: float, num_prefixes: int):
    """按数据集模式下发 random 或 prefix_repetition 命令。"""
    common = [
        "vllm", "bench", "serve",
        "--host", HOST,
        "--port", PORT,
        "--backend", BACKEND,
        "--served-model-name", SERVED_MODEL_NAME,
        "--model", MODEL,
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
    ]

    if dataset == PREFIX_REPETITION_DATASET_NAME:
        # 前缀重复模式:prefix + suffix + output 三段
        # pc_ratio 为该用例的前缀占比(不填时回退全局 PREFIX_REPETITION_PC_RATIO)
        # prefix 和 suffix 都向上取整(可能 prefix+suffix > input_len 1 个 token)
        prefix_len = math.ceil(input_len * pc_ratio)
        suffix_len = math.ceil(input_len * (1 - pc_ratio))
        return common + [
            "--dataset-name", PREFIX_REPETITION_DATASET_NAME,
            "--prefix-repetition-prefix-len", str(prefix_len),
            "--prefix-repetition-suffix-len", str(suffix_len),
            "--prefix-repetition-output-len", str(output_len),
            "--prefix-repetition-num-prefixes", str(num_prefixes),
            IGNORE_EOS,
        ]

    # 默认 random 模式
    return common + [
        "--dataset-name", DATASET_NAME,
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        IGNORE_EOS,
    ]


def _run_subprocess(cmd):
    """执行 vllm bench serve 子进程,返回 stdout。失败抛 CalledProcessError。"""
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        env={**os.environ, 'TERM': 'xterm'},  # 欺骗子进程使其认为有终端,强制输出完整表头
    )
    return result.stdout


def meet_requirements(ttft: float, tpot: float, ttft_max: float, tpot_max: float) -> bool:
    """TTFT 与 TPOT 是否都在阈值内(按原始阈值判定,不加安全余量)。"""
    if ttft == -1 or tpot == -1:
        return False
    return ttft <= ttft_max and tpot <= tpot_max


def _execute_test(cmd, input_len, output_len, concurrency, num_prompts, dataset,
                  vllm_bench_result_file_name, sweep_results_file_name,
                  ttft_max, tpot_max, is_warmup=False, context_suffix=""):
    """执行单次测试,返回 (ttft, tpot, metrics)。失败返回 (-1, -1, {})。"""
    global _bench_err_num
    stage = "预热" if is_warmup else "正式"
    logging.info(
        f"  ┣━ 测并发 {concurrency} ({stage}) il={input_len} ol={output_len} "
        f"np={num_prompts} ds={dataset} ..."
    )
    try:
        output = _run_subprocess(cmd)
        logging.debug(f"{stage}测试原始输出:\n{output}")
        m = _extract_all_metrics(output)

        if m['successful_requests'] <= 0:
            logging.error(f"  ┗━ 并发 {concurrency} 没有成功的请求,测试失败")
            time.sleep(POST_TEST_SLEEP)
            return -1, -1, m

        ttft = select_ttft(m, TTFT_LABEL)
        tpot = select_tpot(m, TPOT_LABEL)

        if not is_warmup:
            write_to_csv(
                [
                    dataset, input_len, output_len, concurrency, num_prompts,
                    m['successful_requests'], m['benchmark_duration'],
                    m['total_input_tokens'], m['total_generated_tokens'],
                    m['req_throughput'], m['output_token_throughput'], m['total_token_throughput'],
                    m['mean_ttft'], m['median_ttft'], m['p99_ttft'],
                    m['mean_tpot'], m['median_tpot'], m['p99_tpot'],
                    m['mean_itl'], m['median_itl'], m['p99_itl'],
                ],
                vllm_bench_result_file_name,
                headers=VLLM_BENCH_HEADERS,
                input_len=input_len,
                output_len=output_len,
                context_suffix=context_suffix,
            )
            passed = meet_requirements(ttft, tpot, ttft_max, tpot_max)
            write_to_csv(
                [dataset, input_len, output_len, concurrency, num_prompts, ttft, tpot, int(passed)],
                sweep_results_file_name,
                headers=SWEEP_HEADERS,
                input_len=input_len,
                output_len=output_len,
                context_suffix=context_suffix,
            )
            logging.info(
                f"  ┗━ 并发 {concurrency} (np={num_prompts}, {dataset}): "
                f"{TTFT_LABEL}={ttft}ms, {TPOT_LABEL}={tpot}ms, "
                f"throughput={m['total_token_throughput']} tok/s"
                + ("，达标" if passed else "，未达标")
            )
            save_perf_log_entry(input_len, output_len, concurrency, num_prompts, dataset, m, output,
                                context_suffix)
        else:
            logging.info(
                f"  ┗━ 预热并发 {concurrency} (np={num_prompts}, {dataset}): "
                f"{TTFT_LABEL}={ttft}ms, {TPOT_LABEL}={tpot}ms"
            )

        time.sleep(POST_TEST_SLEEP)
        return ttft, tpot, m
    except subprocess.CalledProcessError as e:
        # stderr 已合并进 stdout,这里只能从 output 取
        detail = e.output if isinstance(e.output, str) and e.output else e.stderr
        logging.error(f"  ┗━ 并发 {concurrency} 运行出错: {detail}")
        _bench_err_num += 1
        if _bench_err_num < BENCH_MAX_ERRORS:
            return -1, -1, {}
        raise BenchmarkError("请调整参数重新运行") from e


def _formal_test_with_scrape(cmd, input_len, output_len, concurrency, num_prompts, dataset,
                             vllm_bench_result_file_name, sweep_results_file_name,
                             ttft_max, tpot_max, context_suffix):
    """正式测试:前后各抓一次 /metrics,差值算 prefix cache 命中率与投机采样接受率,注入 metrics。"""
    before = _scrape_metrics_snapshot()
    ttft, tpot, metrics = _execute_test(
        cmd, input_len, output_len, concurrency, num_prompts, dataset,
        vllm_bench_result_file_name, sweep_results_file_name,
        ttft_max, tpot_max, is_warmup=False, context_suffix=context_suffix,
    )
    if metrics:
        cache_rate, spec_rate = compute_metrics_rates(before, _scrape_metrics_snapshot())
        metrics['prefix_cache_hit_rate'] = cache_rate
        # fork 版 bench serve 直接打印的本次测试接受率优先于 /metrics 差值
        # (不受指标命名差异与其他流量污染;缺失时为 inf,回退差值口径)
        bench_rate = metrics.get('spec_accept_rate', float('inf'))
        if math.isfinite(bench_rate):
            spec_rate = round(bench_rate, 2)
        metrics['spec_decode_accept_rate'] = spec_rate
    return ttft, tpot, metrics


def run_benchmark_with_metrics(input_len: int, output_len: int, concurrency: int,
                               ttft_max: float, tpot_max: float,
                               vllm_bench_result_file_name: str, sweep_results_file_name: str,
                               dataset: str, num_prompts_ratio: float,
                               pc_ratio: float, num_prefixes: int,
                               warmup_rounds: int = 1) -> Tuple[float, float, dict]:
    """运行 benchmark,提取指标。ENABLE_DOUBLE_RUN 时先用相同命令预热 warmup_rounds 轮,再正式测试。"""
    global _bench_err_num
    num_prompts = compute_num_prompts(concurrency, num_prompts_ratio)
    cmd = _build_bench_cmd(input_len, output_len, concurrency, dataset, num_prompts,
                           pc_ratio, num_prefixes)
    context_suffix = prefix_context_tag(dataset, pc_ratio, num_prefixes)

    try:
        if ENABLE_DOUBLE_RUN:
            for round_idx in range(1, warmup_rounds + 1):
                logging.info(
                    f"开始预热测试 {round_idx}/{warmup_rounds} - 输入长度: {input_len}, "
                    f"输出长度: {output_len}, 并发数: {concurrency}, "
                    f"请求数: {num_prompts}, 数据集: {dataset}"
                )
                warmup_ttft, warmup_tpot, _ = _execute_test(
                    cmd, input_len, output_len, concurrency, num_prompts, dataset,
                    vllm_bench_result_file_name, sweep_results_file_name,
                    ttft_max, tpot_max, is_warmup=True, context_suffix=context_suffix,
                )
                if warmup_ttft == -1 or warmup_tpot == -1:
                    logging.error("预热测试失败，直接返回失败")
                    return -1, -1, {}

            logging.info(
                f"开始正式测试 - 输入长度: {input_len}, 输出长度: {output_len}, "
                f"并发数: {concurrency}, 请求数: {num_prompts}, 数据集: {dataset}"
            )
            return _formal_test_with_scrape(
                cmd, input_len, output_len, concurrency, num_prompts, dataset,
                vllm_bench_result_file_name, sweep_results_file_name,
                ttft_max, tpot_max, context_suffix=context_suffix,
            )
        else:
            return _formal_test_with_scrape(
                cmd, input_len, output_len, concurrency, num_prompts, dataset,
                vllm_bench_result_file_name, sweep_results_file_name,
                ttft_max, tpot_max, context_suffix=context_suffix,
            )
    except Exception as e:
        logging.error(f"测试执行失败: {str(e)}")
        _bench_err_num += 1
        if _bench_err_num < BENCH_MAX_ERRORS:
            return -1, -1, {}
        raise BenchmarkError("请调整参数重新运行") from e


def run_benchmark_with_retry_and_metrics(input_len: int, output_len: int, concurrency: int,
                                         ttft_max: float, tpot_max: float,
                                         vllm_bench_result_file_name: str, sweep_results_file_name: str,
                                         dataset: str, num_prompts_ratio: float,
                                         pc_ratio: float, num_prefixes: int,
                                         warmup_rounds: int = 1,
                                         retries: int = MAX_RETRIES) -> Tuple[float, float, dict]:
    """运行测试,失败则重试,成功即返回。"""
    ttft, tpot, metrics = run_benchmark_with_metrics(
        input_len, output_len, concurrency, ttft_max, tpot_max,
        vllm_bench_result_file_name, sweep_results_file_name,
        dataset, num_prompts_ratio, pc_ratio, num_prefixes, warmup_rounds,
    )
    if ttft != -1 and tpot != -1:
        logging.info(f"测试: {TTFT_LABEL}={ttft}ms, {TPOT_LABEL}={tpot}ms, 并发={concurrency}")
        return ttft, tpot, metrics

    logging.warning("测试失败，服务不可用，开始重试")
    for attempt in range(1, retries):
        ttft, tpot, metrics = run_benchmark_with_metrics(
            input_len, output_len, concurrency, ttft_max, tpot_max,
            vllm_bench_result_file_name, sweep_results_file_name,
            dataset, num_prompts_ratio, pc_ratio, num_prefixes, warmup_rounds,
        )
        if ttft != -1 and tpot != -1:
            logging.info(f"第 {attempt + 1} 次重试: {TTFT_LABEL}={ttft}ms, {TPOT_LABEL}={tpot}ms, 并发={concurrency}")
            return ttft, tpot, metrics
        logging.warning(f"第 {attempt + 1} 次重试失败，服务不可用")
        if attempt < retries - 1:
            time.sleep(RETRY_SLEEP)

    logging.error("所有尝试都失败，返回默认值")
    return -1, -1, {}
