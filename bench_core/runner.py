"""测试入口 - 遍历「用例 × 数据集」,执行固定并发点扫描,写汇总 CSV。"""

import csv
import logging
import os
import time

from .benchmark import reset_bench_error_counter
from .config import (
    BEST_METRICS_HEADERS,
    ENABLE_DOUBLE_RUN,
    IO,
    POINT_METRICS_HEADERS,
    SCRIPT_START_DATE,
    SCRIPT_START_TIME,
    SUMMARY_HEADERS,
    VERSION,
    dataset_prefix_fields,
    resolve_case,
    resolve_warmup_rounds,
)
from .csv_io import get_base_filename, write_to_csv
from .metrics import reset_warnings
from .sweep import point_metrics_row, run_concurrency_sweep


_METRIC_KEYS = [
    'successful_requests', 'benchmark_duration',
    'total_input_tokens', 'total_generated_tokens',
    'req_throughput', 'output_token_throughput', 'total_token_throughput',
    'mean_ttft', 'median_ttft', 'p99_ttft',
    'mean_tpot', 'median_tpot', 'p99_tpot',
    'mean_itl', 'median_itl', 'p99_itl',
]


def _write_summary_csv(summary_results):
    summary_dir = os.path.join(os.getcwd(), "bench", "log", SCRIPT_START_DATE)
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, f"summary_{SCRIPT_START_DATE}_{SCRIPT_START_TIME}.csv")
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADERS)
        for row in summary_results:
            writer.writerow([row[h] for h in SUMMARY_HEADERS])
    logging.info(f"汇总CSV已写入: {summary_file}")


def _write_best_metrics_csv(summary_results):
    best_metrics_dir = os.path.join(os.getcwd(), "bench")
    os.makedirs(best_metrics_dir, exist_ok=True)
    best_metrics_file = os.path.join(
        best_metrics_dir, f"best_metrics_{SCRIPT_START_DATE}_{SCRIPT_START_TIME}.csv"
    )
    with open(best_metrics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(BEST_METRICS_HEADERS)
        for row in summary_results:
            metrics = row.get("metrics", {})
            writer.writerow([
                row["dataset"], row["input_len"], row["output_len"],
                row["pc_ratio"], row["num_prefixes"],
                row["best_concurrency"], row["best_num_prompts"],
            ] + [metrics.get(k, 0) for k in _METRIC_KEYS])
    logging.info(f"最优指标CSV已写入: {best_metrics_file}")


def _write_all_perf_csv(sweeps):
    """把本次运行所有场景的逐并发点关键性能指标汇总成一张表,写到 bench/import_all_perf.csv。"""
    out_dir = os.path.join(os.getcwd(), "bench")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "import_all_perf.csv")
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(POINT_METRICS_HEADERS)
        for sweep in sweeps:
            for p in sweep.points:
                if not p.failed:
                    writer.writerow(
                        point_metrics_row(
                            sweep.dataset, sweep.input_len, sweep.output_len, p,
                            sweep.pc_ratio, sweep.num_prefixes,
                        )
                    )
    logging.info(f"全场景性能汇总CSV已写入: {out_file}")


def run_test_cases():
    """遍历「用例 × 数据集」执行固定并发点扫描,并写汇总 CSV。"""
    logging.info(f"[{VERSION}] 开始 vllm benchmark 固定并发点扫描")
    start_time = time.time()
    summary_results = []
    sweeps = []

    cases = [resolve_case(c) for c in IO]
    total = sum(len(c["datasets"]) for c in cases)
    idx = 0

    for case in cases:
        input_len, output_len = case["input_len"], case["output_len"]
        ttft_max, tpot_max = case["ttft_max"], case["tpot_max"]
        ratio = case["num_prompts_ratio"]
        pc_ratio, num_prefixes = case["pc_ratio"], case["num_prefixes"]

        for dataset in case["datasets"]:
            idx += 1
            # CSV 标识列取值:仅 prefix_repetition 记录前缀参数,random 等留空
            ds_pc, ds_np = dataset_prefix_fields(dataset, pc_ratio, num_prefixes)
            warmup_rounds = resolve_warmup_rounds(case["warmup_rounds"], dataset)
            warmup_info = f"  预热轮数={warmup_rounds}" if ENABLE_DOUBLE_RUN else ""
            logging.info("=" * 72)
            logging.info(
                f"[用例 {idx}/{total}] il={input_len} ol={output_len} ds={dataset}  "
                f"TTFT≤{ttft_max}ms TPOT≤{tpot_max}ms  "
                f"并发点={case['concurrency']}  请求数:并发={ratio:g}:1{warmup_info}"
            )
            logging.info("=" * 72)
            test_start_time = time.time()

            # 每个「用例 × 数据集」重置错误计数与警告缓存
            reset_bench_error_counter()
            reset_warnings()

            result_csv = get_base_filename(
                "vllm_bench_result", input_len, output_len, ttft_max, tpot_max, dataset
            )
            sweep_csv = get_base_filename(
                "sweep_results", input_len, output_len, ttft_max, tpot_max, dataset
            )
            point_metrics_csv = get_base_filename(
                "point_metrics", input_len, output_len, ttft_max, tpot_max, dataset
            )

            sweep = run_concurrency_sweep(
                input_len, output_len, case["concurrency"],
                dataset, ratio, ttft_max, tpot_max,
                result_csv, sweep_csv, point_metrics_csv,
                pc_ratio, num_prefixes, warmup_rounds,
            )
            sweeps.append(sweep)
            best = sweep.best

            test_time = int(time.time() - test_start_time)
            if best:
                logging.info(
                    f"[用例 {idx}/{total}] 完成: 最优并发={best.concurrency} (np={best.num_prompts}), "
                    f"TTFT={best.ttft}ms, TPOT={best.tpot}ms, "
                    f"达标点数={sweep.passed_count}/{len(sweep.points)}, 耗时={test_time}秒"
                )
                best_c, best_np = best.concurrency, best.num_prompts
                ttft, tpot, metrics = best.ttft, best.tpot, best.metrics
            else:
                logging.warning(
                    f"[用例 {idx}/{total}] 完成: 所有并发点都未达标, 无最优并发, 耗时={test_time}秒"
                )
                best_c, best_np = 0, 0
                ttft, tpot, metrics = float('inf'), float('inf'), {}

            summary_results.append({
                "dataset": dataset,
                "input_len": input_len,
                "output_len": output_len,
                "pc_ratio": ds_pc,
                "num_prefixes": ds_np,
                "num_prompts_ratio": ratio,
                "ttft_threshold": ttft_max,
                "tpot_threshold": tpot_max,
                "best_concurrency": best_c,
                "ttft": ttft,
                "tpot": tpot,
                "points_total": len(sweep.points),
                "points_passed": sweep.passed_count,
                "best_num_prompts": best_np,
                "metrics": metrics,
            })

    total_time = int(time.time() - start_time)
    logging.info(f"测试结束，总用时: {total_time}秒")

    _write_summary_csv(summary_results)
    _write_best_metrics_csv(summary_results)
    _write_all_perf_csv(sweeps)
