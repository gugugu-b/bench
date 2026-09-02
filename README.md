# Bench — vLLM 固定并发点扫描压测工具

> 基于 `vllm bench serve`：按配置好的并发列表逐点压测，提取 TTFT / TPOT / 吞吐等指标，
> 在 TTFT / TPOT 阈值内找出**最大达标并发**。

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)](CHANGELOG.md)

---

## 这是什么

大模型推理服务上线 / 性能调优时，经常要回答一个问题：**给定 TTFT / TPOT 阈值，这台服务在各并发档位下表现如何？最大能稳定扛多少并发？**

本工具的做法：

1. 在 `config.IO` 里写好要测的输入输出组合与并发列表（如 `[1, 4, 8, 16, 32, 64, 128]`）；
2. 对每个「用例 × 数据集」组合，逐个并发点调用 `vllm bench serve`（先预热一次，再取正式结果）；
3. 从子进程输出中正则提取 TTFT / TPOT / 吞吐等全部指标，判定该点是否达标；
4. 扫描结束打印横向对比表，**最优并发 = 达标的最大并发**，并落盘 CSV 与 perf_log。

适用场景：**并发容量摸底**、**模型 / 量化方案变更后的回归对比**、**调参前后效果验证**。

---

## 快速开始

### 环境要求

- **Python 3.9+**（本项目仅用标准库，无需 `pip install`）
- 被测环境可运行 `vllm bench serve` 子命令（参考 [vLLM 官方文档](https://docs.vllm.ai/en/latest/benchmarking/)）

### 跑起来

```bash
python run.py
```

先在 `bench_core/config.py` 里改好服务地址（`HOST` / `PORT`）、模型名与测试用例，再运行。

---

## 配置

所有配置集中在 `bench_core/config.py`：

| 配置项                                                  | 含义                                                                                                                                                                        | 默认值                                    |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `IO`                                                  | 测试用例列表，字段见下表                                                                                                                                                    | 见文件                                    |
| `DATASET_MODES`                                       | 全局默认数据集列表，支持`random` / `prefix_repetition`；用例可用 `datasets` 字段单独指定                                                                              | 两者都跑                                  |
| `NUM_PROMPTS_RATIO`                                   | 全局「请求数 : 并发数」比例                                                                                                                                                 | `2`                                     |
| `STOP_ON_BREACH`                                      | 某点未达标时是否提前结束本组合剩余并发点                                                                                                                                    | `False`                                 |
| `DEFAULT_TTFT_MAX` / `DEFAULT_TPOT_MAX`             | 全局阈值默认值 (ms)                                                                                                                                                         | `3000` / `100`                        |
| `TTFT_LABEL` / `TPOT_LABEL`                         | 判定用指标，可选 Mean / Median / P99                                                                                                                                        | `Mean`                                  |
| `HOST` / `PORT` / `SERVED_MODEL_NAME` / `MODEL` | 被测 vLLM 服务信息                                                                                                                                                          | 见文件                                    |
| `PREFIX_REPETITION_PC_RATIO` 等                       | 前缀重复数据集全局默认参数（前缀占比 / 前缀数），可在`IO` 用例里用 `pc_ratio` / `num_prefixes` 按用例覆盖                                                             | `0.9` / `1`                           |
| `ENABLE_DOUBLE_RUN`                                   | 是否开启预热：正式测试前先用相同命令预热若干轮                                                                                                                              | `True`                                  |
| `WARMUP_ROUNDS` | 预热轮数（`ENABLE_DOUBLE_RUN=True` 时生效）；int 为所有数据集统一轮数，dict 按数据集指定（如 `{"random": 1, "prefix_repetition": 4}`）；用例级 `warmup_rounds` 可覆盖 | `{"random": 1, "prefix_repetition": 4}` |
| `ENABLE_METRICS_SCRAPE` 等 | 是否从被测服务 `/metrics`（Prometheus，vLLM 默认与 API 同端口）抓取指标，计算 prefix cache 命中率与投机采样接受率；`METRICS_SCRAPE_PATH` / `METRICS_SCRAPE_TIMEOUT` 控制路径与超时 | `True` / `/metrics` / `5` |
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
| `pc_ratio`                   | 否   | prefix_repetition 模式下 prefix 在输入长度中的占比，缺省用全局`PREFIX_REPETITION_PC_RATIO`；支持输入输出相同、仅占比不同的多组用例 |
| `num_prefixes`               | 否   | prefix_repetition 模式的前缀数，缺省用全局`PREFIX_REPETITION_NUM_PREFIXES`                                                         |
| `datasets`                   | 否   | 该用例要跑的数据集列表，缺省用全局`DATASET_MODES`；只需对比前缀占比等单数据集场景可避免重复跑 random                               |
| `warmup_rounds`              | 否   | 预热轮数：int（该用例统一）或`{dataset: 轮数}`（按数据集），缺省用全局 `WARMUP_ROUNDS`；填 0 表示该场景不预热                    |

请求数 = `ceil(并发 × 比例)`；比例 ≥ 1 时保证不少于并发数。

---

## 输出说明

运行产物统一写入当前工作目录下的 `bench/`：

```
bench/
├── best_metrics_YYYYMMDD_HHMMSS.csv          # 最优并发点的全量指标（每用例×数据集一行）
├── import_all_perf.csv                       # 全场景汇总：所有场景的逐并发点关键性能指标（每次运行重写）
├── log/
│   └── YYYYMMDD/                             # 按运行日期归档
│       ├── summary_YYYYMMDD_HHMMSS.csv       # 汇总：最优并发 / 阈值 / 达标点数等
│       └── context_<il>x<ol>/                # 按场景分目录；prefix_repetition 数据集目录名
│           │                                 #   追加 _pc{占比}_np{前缀数} 后缀（如 context_1024x1024_pc0.9_np1）
│           ├── vllm_bench_result-*.csv       # 每个并发点的全部提取指标（追加写）
│           ├── sweep_results-*.csv           # 每个并发点一行：ttft / tpot / passed
│           └── point_metrics-*.csv           # 每个并发点一行：关键性能指标（见下）
└── perf_log/
    └── <模型名>_<dataset>[_pc{占比}_np{前缀数}]/
        └── il*_ol*_np*_mc*.log               # 原始子进程输出 + 提取的指标（文件名格式固定，供导入使用）
```

- `sweep_results` 的 `passed=1` 表示该点 TTFT 与 TPOT 同时在阈值内；
- `point_metrics` 每个成功并发点一行，标识列为 `dataset` / `input_len` / `output_len` /
  `concurrency` / `pc_ratio` / `num_prefixes`（前缀参数，用于区分输入输出相同但前缀配置不同的
  用例；仅 prefix_repetition 数据集记录真实值，random 等不使用前缀参数的数据集这两列留空，不再
  记录回退默认值 0.9/1），
  指标列为：`mean_ttft` / `mean_tpot`（平均延迟）、
  `output_token_throughput`（生成输出吞吐）、`total_token_throughput`（总吞吐）、
  `benchmark_duration`（总耗时，秒），以及两个单并发归一化指标：
  `output_throughput_per_concurrency`（单并发输出吞吐 = 生成输出吞吐 ÷ 并发数）、
  `decode_throughput_per_concurrency`（单并发 decode 吞吐 = 1000 ÷ 平均 TPOT，
  即单条请求流在 decode 阶段的 token 速率）；
  最后两列来自被测服务 `/metrics` 正式测试前后快照（百分数），按指标前缀自动识别
  vLLM / SGLang 后端：
  - vLLM：`prefix_cache_hit_rate` = Δhits ÷ Δqueries × 100
    （兼容 `vllm:gpu_prefix_cache_*` / `vllm:prefix_cache_*` / `vllm:cpu_prefix_cache_*`）；
    `spec_decode_accept_rate` = Δaccepted ÷ Δdraft × 100
    （`vllm:spec_decode_num_accepted_tokens` / `..._draft_tokens`，均为计数器差值）
  - SGLang：`prefix_cache_hit_rate` = Δcached ÷ Δprompt × 100（token 级命中率）；
    首选 `sglang:cached_tokens_total` / `sglang:prompt_tokens_total` 计数器差值，
    unified 等版本不导出 `cached_tokens_total` 样本时自动退回
    `1 - Δsglang:uncached_prompt_tokens_histogram_sum ÷ Δprompt_tokens_total`
    （SGLang 语义： uncached = prompt − cached；prompt 计数器缺失时分母用
    `sglang:prompt_tokens_histogram_sum`；不使用按 prefill 批次覆盖的
    `sglang:cache_hit_rate` Gauge）；
    `spec_decode_accept_rate` 取测试后快照的 `sglang:spec_accept_rate`
    （SGLang 未暴露接受 token 计数器，该值为服务端 Gauge 而非本次测试差值）；
    多 dp_rank 部署按 dp_rank 配对 `sglang:spec_accept_length`——
    accept_length 经过任何投机批次必然 ≥ 1，length = 0 的 rank 视为空闲，
    其 0 值 rate 不参与平均；无法配对时退回全部序列算术平均；
    ≤ 1 的值按 0-1 比率换算百分数，> 1 视为已是百分数
  抓取失败、服务无该指标或分母为 0 时留空；同名多序列聚合规则：计数器求和、Gauge 取平均；
- `bench/import_all_perf.csv`：上述 point_metrics 的全场景汇总表（列完全相同），
  收录本次运行所有「用例 × 数据集」组合的每个成功并发点，运行结束整体重写；
- `summary` 与 `best_metrics` 同样带 `pc_ratio` / `num_prefixes` 标识列；
  `summary` 的 `points_passed / points_total` 为该组合的达标点数与总点数。

---

## 项目结构

```
.
├── run.py                  # 入口：配置日志后调 run_test_cases()
├── bench_core/
│   ├── __init__.py         # 包导出
│   ├── config.py           # 全部配置常量与用例解析
│   ├── benchmark.py        # 子进程调用 / 指标落盘 / 预热与重试
│   ├── sweep.py            # 固定并发点扫描与结果结构
│   ├── runner.py           # 用例×数据集遍历与汇总 CSV
│   ├── csv_io.py           # CSV 追加写与文件名生成
│   └── metrics.py          # 正则提取与指标选择
├── CHANGELOG.md            # 版本变更记录
└── requirements.txt        # 仅标准库，无第三方依赖
```

---

## 版本历史

见 [CHANGELOG.md](CHANGELOG.md)。当前版本 **v1.1.0**。
