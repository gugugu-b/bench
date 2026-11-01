# Changelog

## v1.4.0 - 2026-11-01

- **可选 SGLang bench**：新增 `BENCH_TOOL`，默认仍为 `"vllm"`；SGLang 使用独立的
  Python、模块入口与请求协议配置，调度、外部预热、固定并发扫描和达标判定共用原流程。
- **双工具数据集映射**：现有 `random` / `prefix_repetition` 用例在 SGLang 下分别转换为
  `random-ids` / `generated-shared-prefix`，也接受两种 SGLang 原生名称。
  显式设置固定采样长度与内部预热请求数 `0`，默认不清空缓存；GSP 前缀组数沿用
  `num_prefixes`（默认 `1`），实际请求总数保持原约定，无法整除时给出配置错误。
- **可选自动选择**：`AUTO_SELECT_BENCH=False` 默认关闭；开启后在运行开始前识别一次
  服务端并选择同名压测工具。识别与性能指标抓取开关独立，失败或证据冲突时告警并回退
  手工配置的工具。
- **bench 日志接受率优先**：直接打印的 `Acceptance rate (%)` 优先；SGLang 服务端与
  SGLang bench 组合支持从 `Accept length:` 按投机参数换算百分数，参数优先来自
  `/server_info`，手工配置兜底。EAGLE / steps=3 / topk=1 / draft tokens=4 时，
  长度 `2.74` 换算约 `58%`；无可用日志值时再回退现有 `/metrics` 计算。
  换算沿用日志原本统计范围，不保证累计接受长度只覆盖当前并发点。
- **统一结果与追溯信息**：兼容 SGLang 的 E2E Latency，避免混入准备阶段、Peak 或
  retokenized 指标。保留原有 CSV 列顺序并追加工具、服务端、实际数据集和接受率来源；
  SGLang 全量指标文件使用 `sglang_bench_result`，perf_log 目录追加 `_sglang`。
- **保留历史运行结果**：`import_all_perf.csv` 改为跨次运行追加，首次创建或空文件时写一次表头。
  兼容与当前表头前缀一致的旧版 CSV，为历史行补齐新增空列；未知或重排等不兼容表头会
  报错并保留原文件，不追加数据。
  perf_log 改为 `bench/perf_log/<SCRIPT_START_STAMP>/<模型名>_<dataset>{前缀后缀}{_sglang}/`
  分运行归档，保留固定日志文件名，运行间不覆盖。
- **微秒级运行时间戳**：`SCRIPT_START_STAMP` 格式改为 `YYYYMMDD_HHMMSS_ffffff`，
  `bench/log/`、perf_log 运行目录、`best_metrics` 与 `summary` 文件名共用该时间戳，
  避免同秒启动时产物路径碰撞。
- **发布验证**：52 项离线测试通过；SGLang bench 实机验证 8 个场景、11 个并发点通过，
  另 2 个 vLLM bench 兼容场景使用隔离 benchmark 启动器验证通过。验证环境原有 vLLM
  全局 CLI 存在 OpenAI `NamespaceTool` 依赖导入错误，未修改全局依赖；默认流程仍要求
  `vllm bench serve` 可用。实机输入 128、输出 32、并发 1 / 2，仅验证功能流程。

## v1.3.4

- 修复：SGLang 后端 `spec_decode_accept_rate` 失真——实测记录 100% / 59.77%，
  而服务端稳态真实值约 93%~95%。根因：`sglang:spec_accept_rate` 是服务端
  **短窗口 Gauge**，原实现在正式测试结束后单次抓取，此时请求已排空，读数退化为
  尾部若干批次的瞬时值——贪心采样（`--temperature 0`）下尾部批次几乎全接受，
  Gauge 读数 1.0，即记录为 100%。现为 SGLang 后端在正式测试**进行中**按
  `SPEC_RATE_SAMPLE_INTERVAL`（`config.py` 新增，默认 5 秒）后台定时采样该
  Gauge 并取平均，无有效采样时退回原单次快照口径；首个采样点延后一个间隔，
  跳过测试初期的 prefill 阶段。vLLM 后端口径不变（计数器差值 + bench serve
  打印值优先；fork 的 bench serve 对 SGLang 后端不打印 `Acceptance rate (%)`，
  该优先级对 SGLang 无影响）。

## v1.3.3

- **`import_all_perf.csv` / `point_metrics` 新增 `mean_e2el` 列**：逐 token 端到端
  延迟均值（毫秒），置于 `benchmark_duration` 之后，取值口径参考 SLO 项目——
  从 bench serve 输出正则提取 `Mean E2EL (ms)`，未打印该指标时留空；
  两表共用 `POINT_METRICS_HEADERS`，`vllm_bench_result` 全量指标表结构不变。
- **bench serve 命令下发 `--percentile-metrics`**：`config.py` 环境配置区新增
  `PERCENTILE_METRICS`（默认 `ttft,tpot,itl,e2el`，两种数据集模式共用）。
  上游 vLLM 默认只统计 `ttft,tpot,itl`，不含 `e2el`——不带该参数时 bench 输出
  没有 E2EL 统计，`mean_e2el` 列将一直为空，两项变更配套生效。

## v1.3.2

- **输出目录按执行时间戳归档**：`bench/log/` 由按运行日期（`YYYYMMDD`）改为按
  执行时间戳（`YYYYMMDD_HHMMSS`，启动时刻）分目录，一次执行一个独立目录，
  同一天多次执行的数据互不覆盖、互不混淆：
  - `config.py` 新增 `SCRIPT_START_STAMP`（单次执行完整时间戳，模块加载时固定），
    `write_to_csv` 与 summary 写入统一改用之；
  - 各 context CSV 与 `summary_<时间戳>.csv` 归入本次执行目录；
  - `best_metrics_<时间戳>.csv` 文件名格式不变（本就含执行时间戳）；
  - `import_all_perf.csv` 与 `perf_log/` 保持原有「每次运行重写、只留最新」语义不变。
- **bench serve 命令下发采样温度**：`config.py` 顶部环境配置区新增 `TEMPERATURE`
  （默认 `0`，random / prefix_repetition 两种模式共用），
  两种数据集模式的命令统一追加 `--temperature` 参数，取值可自行调整。

## v1.3.0

- **配置整理**：`config.py` 环境相关配置集中到文件顶部统一管理——被测服务参数
  （`HOST` / `PORT` / `BACKEND` / `SERVED_MODEL_NAME` / `MODEL` / `IGNORE_EOS`）与
  perf_log 参数（`PERF_LOG_DIR` / `PERF_MODEL_NAME`）原先散落在文件中段，
  现归入顶部「环境配置」分节，切换被测环境只需修改文件顶部一处；
  仅调整位置，变量取值不变，无功能性变更。
- **默认值调整**：`WARMUP_ROUNDS` 中 `prefix_repetition` 的预热轮数由 4 调整为 1，
  即默认值变为 `{"random": 1, "prefix_repetition": 1}`（用例级 `warmup_rounds`
  仍可按需覆盖）。
- 文档：README 配置表与指标口径说明补充服务端前置要求——SGLang 需启动时加
  `--enable-metrics` 才暴露 `/metrics`（vLLM 需 `--enable-prefix-caching` 的
  说明已在 v1.2.0 / v1.2.1 给出）。

## v1.2.1

- 修复：vLLM `/metrics` 计数器样本名自动解析 Prometheus 导出的 `_total` 后缀
  （如 `vllm:prefix_cache_queries` → `vllm:prefix_cache_queries_total`），
  修复厂商 fork（如 vllm_hcu，实测 Qwen3.6-35B-A3B）上 prefix cache 命中率恒为空的问题
  ——原实现只匹配无后缀候选名，与 prometheus_client 的 Counter 标准导出格式不符。
- 修复：`spec_decode_accept_rate` 改为优先采用 bench serve 输出中直接打印的本次测试
  `Acceptance rate (%)`（部分厂商 fork 的 vLLM 打印，如 vllm_hcu，为逐轮精确值），
  输出中无该项时回退 `/metrics` 计数器差值口径（实测该 fork 全部 10 个测试点均可提取，
  如 67.64%；此前该列在 fork 上恒为空，因 `/metrics` 未暴露 spec 计数器时直接放弃）。
- 修复：`pc_ratio` / `num_prefixes` 标识列在 random 等非 prefix_repetition 数据集下
  误记回退默认值（0.9/1）的问题——这两列仅在 prefix_repetition 数据集记录真实值，
  其余数据集留空（`point_metrics` / `import_all_perf` / `summary` / `best_metrics` 四表一致）。
- 修复：测试点重试全部失败或「没有成功的请求」时返回值由 `inf` 改为失败约定 `-1`，
  此前空 metrics 会以成功身份进入 point_metrics 行构造导致 KeyError 中断整个运行。
- 注意：vLLM 服务端需启用 `--enable-prefix-caching`，prefix cache 计数器才会增长，
  否则 `prefix_cache_hit_rate` 为空（实测未开启的部署，计数器恒为 0）。

## v1.2.0

- **prefix cache 命中率与投机采样接受率**：每次正式测试前后各抓取一次被测服务的
  Prometheus `/metrics`（默认与 API 同 host:port），用计数器差值计算两项指标，
  作为 `point_metrics` / `import_all_perf` 表的最后两列（百分数，空值表示不可用）：
  - `prefix_cache_hit_rate` = Δprefix_cache_hits ÷ Δprefix_cache_queries × 100
    （兼容 `vllm:gpu_prefix_cache_*` / `vllm:prefix_cache_*` / `vllm:cpu_prefix_cache_*` 命名）
  - `spec_decode_accept_rate` = Δspec_decode_num_accepted_tokens ÷ Δspec_decode_num_draft_tokens × 100
  - 抓取由 `ENABLE_METRICS_SCRAPE` 控制（默认开），失败只告警一次且不影响测试；
    `METRICS_SCRAPE_PATH` / `METRICS_SCRAPE_TIMEOUT` 可调路径与超时
- **支持 vLLM 与 SGLang 两种服务端**：按 `/metrics` 指标前缀自动识别后端——
  vLLM 用 `vllm:gpu_prefix_cache_hits/queries` 与 `vllm:spec_decode_num_accepted/draft_tokens`
  计数器差值；SGLang 用 `sglang:cached_tokens_total` / `sglang:prompt_tokens_total`
  计数器差值（token 级命中率，不使用按 prefill 批次覆盖的 `sglang:cache_hit_rate` Gauge），
  投机采样接受率取 `sglang:spec_accept_rate` Gauge 测试后快照值
  （SGLang 未暴露接受 token 计数器，≤ 1 的值按比率换算百分数）。
  同名多序列（如多 dp_rank 部署）按指标语义聚合：计数器求和、Gauge 取算术平均。
- **适配 SGLang unified 引擎的真实指标形态**（经 4×dp_rank 服务端真实数据验证）：
  - `cached_tokens_total` 仅有 TYPE 声明、无样本时，cache 命中率自动退回
    `1 - Δuncached_prompt_tokens_histogram_sum ÷ Δprompt_tokens_total` 口径；
  - 投机采样接受率按 dp_rank 配对 `spec_accept_length` 过滤空闲 rank
    （length=0 表示该 rank 从未执行投机批次，0 值 rate 不参与平均），
    避免空闲 rank 拉低整体均值。
- 差值口径为「本次正式测试」，不含预热轮；预热对 cache 的预热效果会体现在正式轮的命中率中
  （SGLang 的 spec_accept_rate 为服务端 Gauge，不含此口径保证）。

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
