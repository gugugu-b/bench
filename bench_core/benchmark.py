"""Benchmark 执行 - 调用选定的压测工具,提取指标,写 CSV/perf_log。"""

import logging
import math
import os
import shlex
import subprocess
import threading
import time
from typing import Tuple

from .config import (
    BENCH_MAX_ERRORS,
    ENABLE_DOUBLE_RUN,
    ENABLE_METRICS_SCRAPE,
    HOST,
    MAX_RETRIES,
    METRICS_SCRAPE_PATH,
    METRICS_SCRAPE_TIMEOUT,
    PERF_LOG_DIR,
    PERF_MODEL_NAME,
    PORT,
    POST_TEST_SLEEP,
    RETRY_SLEEP,
    SCRIPT_START_STAMP,
    SPEC_RATE_SAMPLE_INTERVAL,
    SUBPROCESS_TIMEOUT,
    SWEEP_HEADERS,
    TTFT_LABEL,
    TPOT_LABEL,
    VLLM_BENCH_HEADERS,
    compute_num_prompts,
    prefix_context_tag,
)
from .bench_tools import BenchRuntime, actual_dataset, build_bench_command, resolve_bench_runtime
from .csv_io import write_to_csv
from .metrics import (
    _detect_backend,
    _extract_all_metrics,
    compute_metrics_rates,
    scrape_prometheus_metrics,
    select_ttft,
    select_tpot,
    select_spec_accept_rate,
)


class BenchmarkError(Exception):
    """benchmark 执行过程中的不可恢复错误。"""
    pass


# 子进程连续失败计数,达到 BENCH_MAX_ERRORS 抛 BenchmarkError
_bench_err_num = 0

# /metrics 抓取失败是否已告警过(只告警一次,避免每个测试点刷屏)
_metrics_scrape_warned = False
_bench_runtime = None


def initialize_bench_runtime(runtime: BenchRuntime = None) -> BenchRuntime:
    """每次 run_test_cases 启动时选择一次,各并发点与重试复用同一个选择。"""
    global _bench_runtime, _metrics_scrape_warned
    _bench_runtime = runtime if runtime is not None else resolve_bench_runtime()
    _metrics_scrape_warned = False
    logging.info(
        "压测工具=%s, 服务端=%s, 选择方式=%s",
        _bench_runtime.tool, _bench_runtime.server_engine or "未知", _bench_runtime.selection,
    )
    return _bench_runtime


def get_bench_runtime() -> BenchRuntime:
    """也兼容直接调用单点/扫描 API 的用法。"""
    return _bench_runtime if _bench_runtime is not None else initialize_bench_runtime()


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
                "prefix cache 命中率与投机接受率的 metrics 回退可能不可用"
                "(服务未暴露 /metrics 时可用 ENABLE_METRICS_SCRAPE=False 关闭抓取)"
            )
            _metrics_scrape_warned = True
        return None


def _start_spec_rate_sampler():
    """启动 SGLang spec_accept_rate 后台采样线程,返回 {'stop', 'samples', 'thread'}。

    sglang:spec_accept_rate 为服务端短窗口 Gauge,测试结束排空阶段会退化为
    尾部请求的瞬时值(贪心采样下常读 1.0 即 100%),必须在测试进行中
    定时采样取平均;第一个采样点延后一个间隔,跳过测试启动初期的 prefill 阶段。
    """
    stop_event = threading.Event()
    samples = []

    def _sample_loop():
        while not stop_event.wait(SPEC_RATE_SAMPLE_INTERVAL):
            snapshot = _scrape_metrics_snapshot()
            if snapshot:
                samples.append(snapshot)

    thread = threading.Thread(target=_sample_loop, daemon=True)
    thread.start()
    return {'stop': stop_event, 'samples': samples, 'thread': thread}


def _stop_spec_rate_sampler(sampler):
    """停止采样线程,返回已采集的快照列表。"""
    sampler['stop'].set()
    sampler['thread'].join(timeout=SPEC_RATE_SAMPLE_INTERVAL + METRICS_SCRAPE_TIMEOUT)
    return sampler['samples']


def reset_bench_error_counter():
    """每个测试用例开始时调用,重置错误计数。"""
    global _bench_err_num
    _bench_err_num = 0


def save_perf_log_entry(input_len: int, output_len: int, concurrency: int, num_prompts: int,
                        dataset: str, metrics: dict, raw_output: str, context_suffix: str = "",
                        cmd=None):
    """保存 perf_log 格式的日志条目(原始输出 + 提取的指标)。

    目录: perf_log/<运行时间戳>/<模型名>_<dataset>{context_suffix}[_sglang]/
    文件名: il{input_len}_ol{output_len}_np{np}_mc{concurrency}.log
    文件名格式固定,供外部导入使用;同名用例(如不同 pc_ratio)靠目录后缀区分。
    """
    tool = metrics.get('bench_tool') or get_bench_runtime().tool
    tool_suffix = "_sglang" if tool == "sglang" else ""
    log_dir = os.path.join(
        PERF_LOG_DIR, SCRIPT_START_STAMP,
        f"{PERF_MODEL_NAME}_{dataset}{context_suffix}{tool_suffix}",
    )
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
        f.write(f"Run Stamp: {SCRIPT_START_STAMP}\n")
        f.write(f"Bench Tool: {tool}\n")
        f.write(f"Server Engine: {metrics.get('server_engine', '')}\n")
        f.write(f"Actual Dataset: {metrics.get('actual_dataset', '')}\n")
        if cmd is not None:
            f.write(f"Command: {shlex.join(cmd)}\n")
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
    """通过工具适配层生成命令。"""
    return build_bench_command(
        get_bench_runtime(), input_len, output_len, concurrency,
        dataset, num_prompts, pc_ratio, num_prefixes,
    )


def _run_subprocess(cmd):
    """执行压测子进程,返回 stdout。失败抛 CalledProcessError。"""
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
            return -1, -1, m

        ttft = select_ttft(m, TTFT_LABEL)
        tpot = select_tpot(m, TPOT_LABEL)

        if not is_warmup:
            # 正式轮在补全 metrics 回退和来源后统一落盘;原始输出不暴露给结果对象。
            m['_raw_output'] = output
            passed = meet_requirements(ttft, tpot, ttft_max, tpot_max)
            logging.info(
                f"  ┗━ 并发 {concurrency} (np={num_prompts}, {dataset}): "
                f"{TTFT_LABEL}={ttft}ms, {TPOT_LABEL}={tpot}ms, "
                f"throughput={m['total_token_throughput']} tok/s"
                + ("，达标" if passed else "，未达标")
            )
        else:
            logging.info(
                f"  ┗━ 预热并发 {concurrency} (np={num_prompts}, {dataset}): "
                f"{TTFT_LABEL}={ttft}ms, {TPOT_LABEL}={tpot}ms"
            )

        if is_warmup:
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
    """正式测试:前后各抓一次 /metrics 算 prefix cache 命中率;SGLang 后端另在
    测试进行中定时采样 spec_accept_rate 快照取平均(该 Gauge 为短窗口值,
    测试后单次快照会退化为排空尾部的瞬时值),一并注入 metrics。"""
    runtime = get_bench_runtime()
    before = _scrape_metrics_snapshot()
    sampler = _start_spec_rate_sampler() if _detect_backend(before) == "sglang" else None
    try:
        ttft, tpot, metrics = _execute_test(
            cmd, input_len, output_len, concurrency, num_prompts, dataset,
            vllm_bench_result_file_name, sweep_results_file_name,
            ttft_max, tpot_max, is_warmup=False, context_suffix=context_suffix,
        )
    finally:
        spec_snapshots = _stop_spec_rate_sampler(sampler) if sampler else None
    if metrics and ttft != -1 and tpot != -1:
        raw_output = metrics.pop('_raw_output', '')
        after = _scrape_metrics_snapshot()
        cache_rate, spec_rate = compute_metrics_rates(
            before, after, spec_snapshots,
        )
        metrics['prefix_cache_hit_rate'] = cache_rate
        server_engine = runtime.server_engine or _detect_backend(after) or _detect_backend(before)
        bench_rate, source = select_spec_accept_rate(
            metrics, runtime.tool, server_engine, runtime.server_info,
        )
        if bench_rate != '':
            spec_rate = bench_rate
        elif spec_rate != '':
            source = 'metrics'
        metrics['spec_decode_accept_rate'] = spec_rate
        metrics.update({
            'bench_tool': runtime.tool,
            'server_engine': server_engine,
            'actual_dataset': actual_dataset(dataset, runtime.tool),
            'spec_rate_source': source,
        })
        if not math.isfinite(metrics.get('spec_accept_length', float('inf'))):
            metrics['spec_accept_length'] = ''

        # 在接受率来源确定后写入 CSV 和 perf_log,确保回退值也能追溯。
        write_to_csv(
            [dataset, input_len, output_len, concurrency, num_prompts]
            + [metrics.get(key, '') for key in VLLM_BENCH_HEADERS[5:]],
            vllm_bench_result_file_name, headers=VLLM_BENCH_HEADERS,
            input_len=input_len, output_len=output_len, context_suffix=context_suffix,
        )
        write_to_csv(
            [dataset, input_len, output_len, concurrency, num_prompts, ttft, tpot,
             int(meet_requirements(ttft, tpot, ttft_max, tpot_max))],
            sweep_results_file_name, headers=SWEEP_HEADERS,
            input_len=input_len, output_len=output_len, context_suffix=context_suffix,
        )
        save_perf_log_entry(
            input_len, output_len, concurrency, num_prompts, dataset,
            metrics, raw_output, context_suffix, cmd=cmd,
        )
    time.sleep(POST_TEST_SLEEP)
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
