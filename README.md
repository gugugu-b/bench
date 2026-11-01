# Bench — vLLM / SGLang 固定并发点扫描压测工具

> 默认使用 `vllm bench serve`，可切换 SGLang bench：按配置好的并发列表逐点压测，提取 TTFT / TPOT / 吞吐等指标，
> 在 TTFT / TPOT 阈值内找出**最大达标并发**。

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v1.4.0-blue)](CHANGELOG.md#v140---2026-11-01)
[![Status](https://img.shields.io/badge/Status-Active-success)](CHANGELOG.md)

---

## 这是什么

大模型推理服务上线 / 性能调优时，经常要回答一个问题：**给定 TTFT / TPOT 阈值，这台服务在各并发档位下表现如何？最大能稳定扛多少并发？**

本工具的做法：

1. 在 `config.IO` 里写好要测的输入输出组合与并发列表（如 `[1, 4, 8, 16, 32, 64, 128]`）；
2. 对每个「用例 × 数据集」组合，逐个并发点调用选定的压测工具（按配置预热，再取正式结果）；
3. 从子进程输出中正则提取 TTFT / TPOT / 吞吐等全部指标，判定该点是否达标；
4. 扫描结束打印横向对比表，**最优并发 = 达标的最大并发**，并落盘 CSV 与 perf_log。

适用场景：**并发容量摸底**、**模型 / 量化方案变更后的回归对比**、**调参前后效果验证**。

---

## 快速开始

### 环境要求

- **Python 3.9+**（本项目的调度与解析仅用标准库，无需额外 `pip install`）
- 默认压测环境可运行 `vllm bench serve` 子命令（参考 [vLLM 官方文档](https://docs.vllm.ai/en/latest/benchmarking/)）
- 使用 SGLang bench 时，需要另行准备可运行 SGLang benchmark 模块的 Python 环境；其依赖和 Python 版本要求以安装的 SGLang 版本为准，本仓库不代为安装

### 跑起来

```bash
python run.py
```

先在 `bench_core/config.py` 里改好服务地址（`HOST` / `PORT`）、模型名与测试用例，再运行。

### 选择压测工具

服务端引擎与压测工具分别配置。默认 `BENCH_TOOL = "vllm"`，无论被测服务是 vLLM
还是 SGLang，均继续使用原有 vLLM bench 流程；`BACKEND` 是 vLLM bench 的请求协议参数，
不是压测工具开关。

手动切换 SGLang bench：

```python
BENCH_TOOL = "sglang"
AUTO_SELECT_BENCH = False
SGLANG_BACKEND = "sglang"  # 原生 /generate 请求协议
```

SGLang 子进程默认由 `SGLANG_PYTHON = sys.executable` 启动
`SGLANG_BENCH_MODULE = "sglang.bench_serving"`；可按环境修改 Python 路径，
或将模块改为新版入口 `sglang.benchmark.serving`。

开启 `AUTO_SELECT_BENCH = True` 后，在整次运行开始前只识别一次服务端：vLLM 选择
vLLM bench，SGLang 选择 SGLang bench。识别使用只读信息接口和 `/metrics` 指标前缀，
与 `ENABLE_METRICS_SCRAPE` 独立；不会仅凭端口或 `/v1/models` 猜测。
识别失败或证据冲突时告警并回退 `BENCH_TOOL`。默认关闭自动选择。

### 数据集映射

现有 `IO` 用例无需为切换工具重写数据集名称：

| 用例中的数据集 | vLLM bench 实际数据集 | SGLang bench 实际数据集 |
| --- | --- | --- |
| `random` | `random` | `random-ids` |
| `prefix_repetition` | `prefix_repetition` | `generated-shared-prefix` |

也可直接填写 `random-ids` / `generated-shared-prefix`，按所选工具映射到对应场景。
这些映射表达相近的测试场景，不保证两套工具生成完全相同的输入内容。

SGLang 使用显式 `--random-range-ratio 1` / `--gsp-range-ratio 1` 固定采样长度。
这里固定的是数据生成时的采样长度；`random-ids` 默认仍将生成的 token IDs 解码为文本发送，
实际 token 长度以 bench 结果为准。
共享前缀的组数沿用 `num_prefixes`（默认 `1`），前缀长度为
`ceil(input_len × pc_ratio)`，question 长度为 `ceil(input_len × (1 − pc_ratio))`，
与既有 vLLM 两段取整规则一致，两段相加可能比配置输入长度多 1 个 token；输出长度沿用 `output_len`。
GSP 的每组请求数为 `num_prompts / num_prefixes`；无法整除时明确报配置错误。
SGLang 内部预热固定为 `--warmup-requests 0`，预热统一由本工具的 `WARMUP_ROUNDS`
控制，默认不清空服务端缓存。

---

## 配置

所有配置集中在 `bench_core/config.py`，环境相关配置（被测服务信息、perf_log 输出）统一放在文件顶部「环境配置」分节，切换环境只需改顶部一处：

| 配置项                                                  | 含义                                                                                                                                                                        | 默认值                                    |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `IO`                                                  | 测试用例列表，字段见下表                                                                                                                                                    | 见文件                                    |
| `DATASET_MODES`                                       | 全局默认数据集列表，支持 `random` / `prefix_repetition` 及对应 SGLang 名称；用例可用 `datasets` 字段单独指定 | 两者都跑 |
| `BENCH_TOOL` / `AUTO_SELECT_BENCH` | 手工选择压测工具 / 是否按服务端自动选择 | `"vllm"` / `False` |
| `BENCH_DETECT_TIMEOUT` | 每个服务端识别请求的超时（秒），与性能指标抓取开关独立 | `3` |
| `BACKEND` / `SGLANG_BACKEND` | 分别传给 vLLM / SGLang bench 的请求协议参数 | `"vllm"` / `"sglang"` |
| `SGLANG_PYTHON` / `SGLANG_BENCH_MODULE` | SGLang bench 使用的 Python 与模块入口 | `sys.executable` / `"sglang.bench_serving"` |
| `SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS` / `SGLANG_SPECULATIVE_NUM_STEPS` | `Accept length:` 换算的手工兜底参数；优先使用服务端 `/server_info` 中的投机参数 | `4` / `3` |
| `NUM_PROMPTS_RATIO`                                   | 全局「请求数 : 并发数」比例                                                                                                                                                 | `2`                                     |
| `STOP_ON_BREACH`                                      | 某点未达标时是否提前结束本组合剩余并发点                                                                                                                                    | `False`                                 |
| `DEFAULT_TTFT_MAX` / `DEFAULT_TPOT_MAX`             | 全局阈值默认值 (ms)                                                                                                                                                         | `3000` / `100`                        |
| `TTFT_LABEL` / `TPOT_LABEL`                         | 判定用指标，可选 Mean / Median / P99                                                                                                                                        | `Mean`                                  |
| `HOST` / `PORT` / `SERVED_MODEL_NAME` / `MODEL` | 被测推理服务信息 | 见文件 |
| `TEMPERATURE`                                         | bench serve 采样温度（random / prefix_repetition 两种模式共用）                                                                                                             | `0`                                     |
| `PERCENTILE_METRICS`                                  | vLLM bench `--percentile-metrics` 值，含 `e2el` 才会输出端到端时延统计；SGLang 从自身结果中的 E2E Latency 读取 | `ttft,tpot,itl,e2el` |
| `PREFIX_REPETITION_PC_RATIO` 等                       | 前缀重复数据集全局默认参数（前缀占比 / 前缀数），可在`IO` 用例里用 `pc_ratio` / `num_prefixes` 按用例覆盖                                                             | `0.9` / `1`                           |
| `ENABLE_DOUBLE_RUN`                                   | 是否开启预热：正式测试前先用相同命令预热若干轮                                                                                                                              | `True`                                  |
| `WARMUP_ROUNDS` | 预热轮数（`ENABLE_DOUBLE_RUN=True` 时生效）；int 为所有数据集统一轮数，dict 按数据集指定（如 `{"random": 1, "prefix_repetition": 4}`）；用例级 `warmup_rounds` 可覆盖 | `{"random": 1, "prefix_repetition": 1}` |
| `ENABLE_METRICS_SCRAPE` 等 | 是否从被测服务 `/metrics`（Prometheus）抓取指标，计算 prefix cache 命中率与投机采样接受率；`METRICS_SCRAPE_PATH` / `METRICS_SCRAPE_TIMEOUT` 控制路径与超时，`SPEC_RATE_SAMPLE_INTERVAL` 控制 SGLang 接受率测试中采样间隔（短窗口 Gauge，须稳态期间多次采样取平均）。**服务端要求**：vLLM 默认与 API 同端口暴露（需 `--enable-prefix-caching` 才有 cache 数据）；SGLang 需启动时加 `--enable-metrics` | `True` / `/metrics` / `5` / `5` |
| `MAX_RETRIES` / `BENCH_MAX_ERRORS`                  | 失败重试次数 / 连续失败上限                                                                                                                                                 | `2` / `3`                             |
| `SUBPROCESS_TIMEOUT`                                  | 单次子进程超时（秒）                                                                                                                                                        | `3600`                                  |
| `PERF_LOG_DIR`                                        | perf_log 输出目录                                                                                                                                                           | `./bench/perf_log`                      |

`IO` 中每个用例的字段：

| 字段                           | 必填 | 说明                                                                                                                                 |
| ------------------------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `input_len` / `output_len` | 是   | 输入 / 输出 token 长度                                                                                                               |
| `concurrency`                | 是   | 该组合要跑的并发点列表（写单个 int 等价于`[int]`）                                                                                 |
| `num_prompts_ratio`          | 否   | 「请求数 : 并发数」比例，缺省用全局`NUM_PROMPTS_RATIO`                                                                             |
| `ttft_max` / `tpot_max`    | 否   | 该用例的阈值 (ms)，缺省用全局默认值                                                                                                  |
| `pc_ratio`                   | 否   | 共享前缀场景下 prefix 在输入长度中的占比，缺省用全局 `PREFIX_REPETITION_PC_RATIO`；支持输入输出相同、仅占比不同的多组用例 |
| `num_prefixes`               | 否   | 共享前缀场景的前缀数（SGLang 为组数），缺省用全局 `PREFIX_REPETITION_NUM_PREFIXES`，默认 `1` |
| `datasets`                   | 否   | 该用例要跑的数据集列表，缺省用全局`DATASET_MODES`；只需对比前缀占比等单数据集场景可避免重复跑 random                               |
| `warmup_rounds`              | 否   | 预热轮数：int（该用例统一）或`{dataset: 轮数}`（按数据集），缺省用全局 `WARMUP_ROUNDS`；填 0 表示该场景不预热                    |

请求数 = `ceil(并发 × 比例)`；比例 ≥ 1 时保证不少于并发数。

---

## 输出说明

运行产物统一写入当前工作目录下的 `bench/`；`log/` 和 `perf_log/` 按执行时间戳
（`YYYYMMDD_HHMMSS_ffffff`，启动时刻，末尾六位为微秒）分目录，避免同一秒内启动的
多次执行互相覆盖；`best_metrics` 文件名与 `summary` 共用该时间戳：

```
bench/
├── best_metrics_YYYYMMDD_HHMMSS_ffffff.csv   # 最优并发点的全量指标（每用例×数据集一行）
├── import_all_perf.csv                       # 跨次运行追加：所有场景的逐并发点关键性能指标
├── log/
│   └── YYYYMMDD_HHMMSS_ffffff/               # 按执行时间戳归档，一次执行一个目录
│       ├── summary_YYYYMMDD_HHMMSS_ffffff.csv # 汇总：最优并发 / 阈值 / 达标点数等
│       └── context_<il>x<ol>/                # 按场景分目录；共享前缀场景目录名
│           │                                 #   追加 _pc{占比}_np{前缀数} 后缀（如 context_1024x1024_pc0.9_np1）
│           ├── {vllm,sglang}_bench_result-*.csv # 每个并发点的全部提取指标（追加写）
│           ├── sweep_results-*.csv           # 每个并发点一行：ttft / tpot / passed
│           └── point_metrics-*.csv           # 每个并发点一行：关键性能指标（见下）
└── perf_log/
    └── YYYYMMDD_HHMMSS_ffffff/
        └── <模型名>_<dataset>[_pc{占比}_np{前缀数}][_sglang]/
            └── il*_ol*_np*_mc*.log           # 原始子进程输出 + 提取的指标（固定文件名，运行间不覆盖）
```

- `sweep_results` 的 `passed=1` 表示该点 TTFT 与 TPOT 同时在阈值内；
- `point_metrics` 每个成功并发点一行，标识列为 `dataset` / `input_len` / `output_len` /
  `concurrency` / `pc_ratio` / `num_prefixes`（前缀参数，用于区分输入输出相同但前缀配置不同的
  用例；仅共享前缀场景记录真实值，random 等不使用前缀参数的数据集这两列留空，不再
  记录回退默认值 0.9/1），
  指标列为：`mean_ttft` / `mean_tpot`（平均延迟）、
  `output_token_throughput`（生成输出吞吐）、`total_token_throughput`（总吞吐）、
  `benchmark_duration`（总耗时，秒）、
  `mean_e2el`（请求端到端延迟均值，毫秒；从 vLLM 的 `Mean E2EL` 或 SGLang 的
  `Mean E2E Latency` 提取，未打印时留空），
  以及两个单并发归一化指标：
  `output_throughput_per_concurrency`（单并发输出吞吐 = 生成输出吞吐 ÷ 并发数）、
  `decode_throughput_per_concurrency`（单并发 decode 吞吐 = 1000 ÷ 平均 TPOT，
  即单条请求流在 decode 阶段的 token 速率）；
- `spec_decode_accept_rate`（投机采样接受率，百分数）：**bench 日志优先，无可用值时再回退
  `/metrics`**。先使用日志直接打印的 `Acceptance rate (%)`；SGLang 服务端配合
  SGLang bench 的 EAGLE 场景，也可从 `Accept length:` 换算：
  `接受率 = (Accept length − 1) / (num_draft_tokens − 1) × 100%`。
  优先使用服务端 `/server_info` 的投机参数：有草稿数量时用 `num_draft_tokens − 1`
  作为分母，未提供草稿数量时用 `num_steps`。信息中没有相关投机参数时，使用
  `SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS` / `SGLANG_SPECULATIVE_NUM_STEPS` 手工配置兜底。
  服务端明确指定其他算法、参数无效或换算结果超出 0–100% 时，不套用该公式。
  对 EAGLE / steps=3 / topk=1 / draft tokens=4，`Accept length: 2.74` 对应约 `58%`。
  该换算保留日志原本的统计范围；部分版本的接受长度是服务端累计值，不保证只属于当前并发点。
  参数或日志无可用值时使用下述 `/metrics` 口径，两者都不可用则留空；
- `prefix_cache_hit_rate`（prefix cache 命中率，百分数）：只能来自被测服务 `/metrics`
  正式测试前后快照，按指标前缀自动识别 vLLM / SGLang 后端：
  - vLLM：`prefix_cache_hit_rate` = Δhits ÷ Δqueries × 100
    （兼容 `vllm:gpu_prefix_cache_*` / `vllm:prefix_cache_*` / `vllm:cpu_prefix_cache_*`
    三组候选名，计数器样本名的 `_total` 后缀自动解析）；
    回退口径 `spec_decode_accept_rate` = Δaccepted ÷ Δdraft × 100
    （`vllm:spec_decode_num_accepted_tokens` / `..._draft_tokens`，均为计数器差值）；
    前提：vLLM 服务端需启用 `--enable-prefix-caching`，否则 cache 计数器不增长，
    该列为空
  - SGLang：需在服务端启动参数中加 `--enable-metrics` 开启 `/metrics` 暴露；
    `prefix_cache_hit_rate` = Δcached ÷ Δprompt × 100（token 级命中率）；
    首选 `sglang:cached_tokens_total` / `sglang:prompt_tokens_total` 计数器差值，
    unified 等版本不导出 `cached_tokens_total` 样本时自动退回
    `1 - Δsglang:uncached_prompt_tokens_histogram_sum ÷ Δprompt_tokens_total`
    （SGLang 语义： uncached = prompt − cached；prompt 计数器缺失时分母用
    `sglang:prompt_tokens_histogram_sum`；不使用按 prefill 批次覆盖的
    `sglang:cache_hit_rate` Gauge）；
    回退口径 `spec_decode_accept_rate` 取正式测试**进行中**按 `SPEC_RATE_SAMPLE_INTERVAL`
    （默认 5 秒）定时采样的 `sglang:spec_accept_rate` Gauge 平均值
    （SGLang 未暴露接受 token 计数器，只能取服务端 Gauge；该 Gauge 为短窗口值，
    测试结束后的单次快照在请求排空阶段会退化为尾部瞬时值——贪心采样下常读
    100%，故须稳态期间多次采样取平均；无有效采样时退回测试后单次快照）；
    多 dp_rank 部署按 dp_rank 配对 `sglang:spec_accept_length`——
    accept_length 经过任何投机批次必然 ≥ 1，length = 0 的 rank 视为空闲，
    其 0 值 rate 不参与平均；无法配对时退回全部序列算术平均；
    ≤ 1 的值按 0-1 比率换算百分数，> 1 视为已是百分数
  抓取失败、服务无该指标或分母为 0 时留空；同名多序列聚合规则：计数器求和、Gauge 取平均；
- `bench/import_all_perf.csv`：上述 point_metrics 的全场景汇总表（列完全相同），
  每次运行结束将本次所有「用例 × 数据集」组合的成功并发点追加到已有文件，保留历史运行数据；
  仅在文件首次创建或文件为空时写一次表头。
  已有旧版表头若与当前表头的前缀完全一致，则升级表头并为历史行补齐末尾空列后追加；
  未知、重排等不兼容表头会明确报错并保留原文件，不追加数据；
- `point_metrics` / `import_all_perf` / 全量 `{vllm,sglang}_bench_result` 保留原有列顺序，
  末尾追加 `bench_tool`（实际压测工具）、`server_engine`（服务端类型，未知时留空）、
  `actual_dataset`（传给 CLI 的实际数据集名）、`spec_accept_length`（原始接受长度，缺失时留空）、
  `spec_rate_source`（接受率来源：`bench_acceptance_rate` / `bench_accept_length` / `metrics`，不可用时留空）；
- `summary` 与 `best_metrics` 同样带 `pc_ratio` / `num_prefixes` 标识列；
  末尾追加 `bench_tool` / `server_engine` / `actual_dataset`；
  `summary` 的 `points_passed / points_total` 为该组合的达标点数与总点数；
- perf_log 在 `PERF_LOG_DIR` 下增加本次 `SCRIPT_START_STAMP` 目录，完整路径为
  `bench/perf_log/<SCRIPT_START_STAMP>/<模型名>_<dataset>[_pc{占比}_np{前缀数}][_sglang]/`。
  `il*_ol*_np*_mc*.log` 文件名格式保持固定，SGLang bench 的场景目录追加 `_sglang`，
  相同场景在不同运行中的日志分别归档。

---

## 项目结构

```
.
├── run.py                  # 入口：配置日志后调 run_test_cases()
├── bench_core/
│   ├── __init__.py         # 包导出
│   ├── config.py           # 全部配置常量与用例解析
│   ├── bench_tools.py      # 工具选择 / 服务端识别 / 数据集映射 / 命令构造
│   ├── benchmark.py        # 子进程调用 / 指标落盘 / 预热与重试
│   ├── sweep.py            # 固定并发点扫描与结果结构
│   ├── runner.py           # 用例×数据集遍历与汇总 CSV
│   ├── csv_io.py           # CSV 追加写与文件名生成
│   └── metrics.py          # 双工具指标提取 / 接受率换算 / Prometheus 指标
├── CHANGELOG.md            # 版本变更记录
└── requirements.txt        # 仅标准库，无第三方依赖
```

---

## 验证

运行离线测试，不需要启动推理服务或安装两套压测工具：

```bash
python3 -B -m unittest discover -s tests -v
```

v1.4.0（2026-11-01）发布验证结果：

- **52 项离线测试通过**，覆盖工具选择、数据集命令、指标解析、接受率回退和跨次结果保留。
- **SGLang bench 实机验证 8 个场景、11 个并发点通过**；另有 **2 个 vLLM bench 兼容场景**
  通过隔离 benchmark 启动器完成验证。
- 验证环境原有 vLLM 全局 CLI 存在 OpenAI `NamespaceTool` 依赖导入错误，兼容验证绕过了该入口；
  项目未修改全局依赖。发布版默认流程仍要求测试环境的 `vllm bench serve` 命令本身可用。
- 实机验证采用输入长度 128、输出长度 32、并发 1 / 2，用于检查功能流程，不能作为生产容量结论。

---

## 版本历史

见 [CHANGELOG.md](CHANGELOG.md)。当前版本 **v1.4.0**，发布日期 **2026-11-01**。
