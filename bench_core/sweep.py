"""固定并发点扫描 - 按配置好的并发列表逐点测试,不做自适应搜索。

每个 (input_len, output_len, dataset) 组合把 concurrency 列表里每个点都跑一遍,
记录 TTFT / TPOT 与是否达标;最优并发 = 达标的最大并发。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .benchmark import meet_requirements, run_benchmark_with_retry_and_metrics
from .config import (
    POINT_METRICS_HEADERS,
    STOP_ON_BREACH,
    TTFT_LABEL,
    TPOT_LABEL,
    compute_num_prompts,
    prefix_context_tag,
)
from .csv_io import write_to_csv


@dataclass
class PointResult:
    """单个并发点的测试结果。"""
    concurrency: int
    num_prompts: int
    ttft: float
    tpot: float
    metrics: dict = field(default_factory=dict)
    passed: bool = False
    failed: bool = False   # 子进程崩溃 / 无成功请求

    @property
    def output_token_throughput(self) -> float:
        return self.metrics.get('output_token_throughput', 0)

    @property
    def total_token_throughput(self) -> float:
        return self.metrics.get('total_token_throughput', 0)


@dataclass
class SweepResult:
    """一个 (input_len, output_len, dataset) 组合的扫描结果。"""
    input_len: int
    output_len: int
    dataset: str
    num_prompts_ratio: float
    pc_ratio: float
    num_prefixes: int
    ttft_max: float
    tpot_max: float
    points: List[PointResult] = field(default_factory=list)

    @property
    def passed_points(self) -> List[PointResult]:
        return [p for p in self.points if p.passed]

    @property
    def best(self) -> Optional[PointResult]:
        """达标的最大并发点;一个都不达标时返回 None。"""
        passed = self.passed_points
        return max(passed, key=lambda p: p.concurrency) if passed else None

    @property
    def passed_count(self) -> int:
        return len(self.passed_points)


def point_metrics_row(dataset: str, input_len: int, output_len: int, point: PointResult,
                      pc_ratio: float, num_prefixes: int) -> list:
    """单个并发点的 point_metrics 表行(含两个单并发归一化吞吐列)。"""
    m = point.metrics
    return [
        dataset, input_len, output_len, point.concurrency, pc_ratio, num_prefixes,
        m['mean_ttft'], m['mean_tpot'],
        m['output_token_throughput'], m['total_token_throughput'],
        m['benchmark_duration'],
        m['output_token_throughput'] / point.concurrency,
        # 单并发 decode 吞吐 = 1000/平均TPOT(ms),单条请求流的 decode 速率
        (1000 / m['mean_tpot']) if m['mean_tpot'] > 0 else 0.0,
    ]


def run_concurrency_sweep(input_len: int, output_len: int, concurrency_list: List[int],
                          dataset: str, num_prompts_ratio: float,
                          ttft_max: float, tpot_max: float,
                          vllm_bench_result_file_name: str,
                          sweep_results_file_name: str,
                          point_metrics_file_name: str,
                          pc_ratio: float, num_prefixes: int,
                          warmup_rounds: int = 1) -> SweepResult:
    """按给定并发列表逐点测试,返回 SweepResult。

    concurrency_list 里重复的值只测一次(结果复用缓存)。
    STOP_ON_BREACH=True 时,某点突破阈值就结束本组合剩余的并发点。
    """
    result = SweepResult(
        input_len=input_len,
        output_len=output_len,
        dataset=dataset,
        num_prompts_ratio=num_prompts_ratio,
        pc_ratio=pc_ratio,
        num_prefixes=num_prefixes,
        ttft_max=ttft_max,
        tpot_max=tpot_max,
    )
    # 去重并保持配置里的顺序
    ordered = list(dict.fromkeys(int(c) for c in concurrency_list))
    total_points = len(ordered)
    context_suffix = prefix_context_tag(dataset, pc_ratio, num_prefixes)

    logging.info(
        f"并发点扫描: {ordered} (共 {total_points} 点), 请求数:并发 = {num_prompts_ratio:g}:1, "
        f"数据集={dataset}{context_suffix}, 阈值: {TTFT_LABEL}≤{ttft_max}ms / {TPOT_LABEL}≤{tpot_max}ms"
    )

    for idx, concurrency in enumerate(ordered, 1):
        num_prompts = compute_num_prompts(concurrency, num_prompts_ratio)
        point_start = time.time()

        ttft, tpot, metrics = run_benchmark_with_retry_and_metrics(
            input_len, output_len, concurrency, ttft_max, tpot_max,
            vllm_bench_result_file_name, sweep_results_file_name,
            dataset, num_prompts_ratio, pc_ratio, num_prefixes, warmup_rounds,
        )

        failed = ttft == -1 or tpot == -1
        point = PointResult(
            concurrency=concurrency,
            num_prompts=num_prompts,
            ttft=ttft,
            tpot=tpot,
            metrics=metrics,
            passed=meet_requirements(ttft, tpot, ttft_max, tpot_max),
            failed=failed,
        )
        result.points.append(point)

        if not failed:
            write_to_csv(
                point_metrics_row(dataset, input_len, output_len, point, pc_ratio, num_prefixes),
                point_metrics_file_name,
                headers=POINT_METRICS_HEADERS,
                input_len=input_len,
                output_len=output_len,
                context_suffix=context_suffix,
            )

        elapsed = int(time.time() - point_start)
        if failed:
            logging.warning(
                f"[{idx}/{total_points}] 并发 {concurrency} (np={num_prompts}) 测试失败, 耗时 {elapsed}s"
            )
        else:
            logging.info(
                f"[{idx}/{total_points}] 并发 {concurrency} (np={num_prompts}): "
                f"{TTFT_LABEL}={ttft}ms, {TPOT_LABEL}={tpot}ms, "
                f"输出吞吐={point.output_token_throughput} tok/s, "
                f"总吞吐={point.total_token_throughput} tok/s, "
                f"达标={'是' if point.passed else '否'}, 耗时 {elapsed}s"
            )

        if not point.passed and STOP_ON_BREACH:
            logging.warning(
                f"并发 {concurrency} 未达标且 STOP_ON_BREACH=True，"
                f"提前结束本组合剩余并发点: {ordered[idx:]}"
            )
            break

    _log_sweep_summary(result)
    return result


def _log_sweep_summary(result: SweepResult):
    """扫描结束后打印该组合的横向对比表。"""
    logging.info("-" * 72)
    logging.info(
        f"扫描汇总 il={result.input_len} ol={result.output_len} "
        f"ds={result.dataset}{prefix_context_tag(result.dataset, result.pc_ratio, result.num_prefixes)} "
        f"阈值: TTFT≤{result.ttft_max}ms TPOT≤{result.tpot_max}ms"
    )
    header = f"{'并发':>6} | {'请求数':>7} | {TTFT_LABEL:>12} | {TPOT_LABEL:>12} | {'总吞吐(tok/s)':>14} | 达标"
    logging.info(header)
    logging.info("-" * len(header))
    for p in result.points:
        if p.failed:
            logging.info(f"{p.concurrency:>6} | {p.num_prompts:>7} | {'FAILED':>12} | {'FAILED':>12} | {'-':>14} | -")
        else:
            logging.info(
                f"{p.concurrency:>6} | {p.num_prompts:>7} | {p.ttft:>12.2f} | {p.tpot:>12.2f} | "
                f"{p.total_token_throughput:>14.2f} | {'是' if p.passed else '否'}"
            )
    best = result.best
    if best:
        logging.info(
            f"结论: 最优并发 = {best.concurrency} (np={best.num_prompts}), "
            f"{TTFT_LABEL}={best.ttft}ms, {TPOT_LABEL}={best.tpot}ms, "
            f"总吞吐={best.total_token_throughput} tok/s"
        )
    else:
        logging.warning("结论: 所有并发点都未达标，无最优并发")
    logging.info("-" * 72)
