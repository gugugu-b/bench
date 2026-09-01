# Changelog

## v1.0.0 - 首个正式版本

> 此前项目以 v1.x / v2.x 内部迭代（自适应并发搜索、包更名为 `bench_core`、移除 SLO 命名等），
> 现版本号统一归零，以当前形态作为首个正式版本发布；历史细节见 git 提交记录。

### 功能

- **固定并发点扫描**：每个 (input_len, output_len, dataset) 组合按 `config.IO` 中显式配置的
  并发列表逐点执行 `vllm bench serve`，不做自适应搜索；列表中重复的并发值只测一次。
- **多用例 × 多数据集**：`DATASET_MODES` 支持 `random` 与 `prefix_repetition`
  （前缀占比 / 前缀数可配），按「用例 × 数据集」全组合遍历。
- **请求数比例**：`num_prompts = ceil(并发 × ratio)`，比例可用全局 `NUM_PROMPTS_RATIO`
  或按用例覆盖；比例 ≥ 1 时保证请求数不少于并发数。
- **阈值判定与最优并发**：按用例级（缺省回退全局）TTFT / TPOT 阈值逐点判定是否达标，
  最优并发 = 达标的最大并发；`STOP_ON_BREACH = True` 可在某点未达标后提前结束本组合。
- **预热与重试**：`ENABLE_DOUBLE_RUN` 开启时第一次预热、第二次计为正式结果；
  子进程失败自动重试（`MAX_RETRIES`），连续失败超过 `BENCH_MAX_ERRORS` 抛错终止。
- **指标提取**：正则提取 successful_requests / benchmark_duration / tokens /
  throughput 及 TTFT / TPOT / ITL 的 mean / median / p99；判定指标可选 Mean / Median / P99。
- **结果落盘**（详见 README「输出说明」）：
  - `bench/log/<日期>/context_<il>x<ol>/vllm_bench_result-*.csv`：每个并发点的全量指标（追加写）；
  - `bench/log/<日期>/context_<il>x<ol>/sweep_results-*.csv`：每个并发点一行（ttft / tpot / passed）；
  - `bench/log/<日期>/summary_*.csv`：每个「用例 × 数据集」一行的汇总；
  - `bench/best_metrics_*.csv`：最优并发点的完整指标；
  - `bench/perf_log/<模型名>/*.log`：原始子进程输出 + 提取的指标。
- 控制台实时输出每点结果，扫描结束打印横向对比表与结论。

### 约定

- 仅依赖 Python 标准库（≥ 3.9）；被测侧需提供 `vllm bench serve` CLI。
- 运行时产物统一写入 `bench/` 目录，已被 `.gitignore` 忽略。
