# Changelog

## v1.1.0

- **按用例自定义前缀重复参数**：`IO` 用例新增可选字段 `pc_ratio`（prefix 在输入长度中的占比）
  与 `num_prefixes`（前缀数），缺省回退全局 `PREFIX_REPETITION_PC_RATIO` / `PREFIX_REPETITION_NUM_PREFIXES`；
  支持输入输出完全相同、仅前缀参数不同的多组用例（默认配置即含 1024×1024 的 pc 10% / 90% 两组）。
- **预热轮数可配**：`ENABLE_DOUBLE_RUN=True` 时，正式测试前用完全相同的命令先预热
  `WARMUP_ROUNDS` 轮。全局 `WARMUP_ROUNDS` 可为 int（所有数据集统一轮数）或 dict
  （按数据集指定，如 `{"random": 1, "prefix_repetition": 4}`，默认即此值）；
  用例级 `warmup_rounds` 字段可覆盖（写法相同，填 0 表示该场景不预热）。
- **数据集按用例指定**：`IO` 用例新增可选 `datasets` 字段，指定该用例要跑的数据集列表，
  缺省回退全局 `DATASET_MODES`。
- **目录与表格区分前缀参数**：prefix_repetition 数据集的上下文目录由 `context_<il>x<ol>`
  改为 `context_<il>x<ol>_pc{占比}_np{前缀数}`（random 数据集目录名不变）；
  perf_log 目录由 `<模型名>_<dataset>` 改为 `<模型名>_<dataset>_pc{占比}_np{前缀数}`
  （文件名 `il*_ol*_np*_mc*.log` 格式固定不变，供外部导入；避免输入输出相同、
  仅前缀参数不同的用例日志互相覆盖）；
  `point_metrics` / `import_all_perf` / `summary` / `best_metrics` 均新增
  `pc_ratio`、`num_prefixes` 标识列。
- 注：输入输出相同的多组用例若都跑 random 会重复执行并追加到同一文件；
  只想对比前缀占比时，给这些用例加 `"datasets": ["prefix_repetition"]` 即可。

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
  - `bench/log/<日期>/context_<il>x<ol>/point_metrics-*.csv`：每个成功并发点一行的关键性能指标
    （平均 TTFT/TPOT、生成输出吞吐、总吞吐、总耗时、单并发输出吞吐 = 生成输出吞吐 ÷ 并发数、
    单并发 decode 吞吐 = 1000 ÷ 平均 TPOT）；
  - `bench/import_all_perf.csv`：全场景汇总表，收录本次运行所有「用例 × 数据集」组合的
    逐并发点关键性能指标（列与 point_metrics 相同），运行结束整体重写；
  - `bench/log/<日期>/summary_*.csv`：每个「用例 × 数据集」一行的汇总；
  - `bench/best_metrics_*.csv`：最优并发点的完整指标；
  - `bench/perf_log/<模型名>_<dataset>/*.log`：原始子进程输出 + 提取的指标
    （文件名 `il*_ol*_np*_mc*.log`，数据集并入目录名）；
- 控制台实时输出每点结果，扫描结束打印横向对比表与结论。

### 约定

- 仅依赖 Python 标准库（≥ 3.9）；被测侧需提供 `vllm bench serve` CLI。
- 运行时产物统一写入 `bench/` 目录，已被 `.gitignore` 忽略。
