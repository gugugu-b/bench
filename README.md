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

| 配置项 | 含义 | 默认值 |
|---|---|---|
| `IO` | 测试用例列表，字段见下表 | 见文件 |
| `DATASET_MODES` | 数据集列表，支持 `random` / `prefix_repetition` | 两者都跑 |
| `NUM_PROMPTS_RATIO` | 全局「请求数 : 并发数」比例 | `2` |
| `STOP_ON_BREACH` | 某点未达标时是否提前结束本组合剩余并发点 | `False` |
| `DEFAULT_TTFT_MAX` / `DEFAULT_TPOT_MAX` | 全局阈值默认值 (ms) | `3000` / `100` |
| `TTFT_LABEL` / `TPOT_LABEL` | 判定用指标，可选 Mean / Median / P99 | `Mean` |
| `HOST` / `PORT` / `SERVED_MODEL_NAME` / `MODEL` | 被测 vLLM 服务信息 | 见文件 |
| `PREFIX_REPETITION_PC_RATIO` 等 | 前缀重复数据集参数（前缀占比 / 前缀数） | `0.9` / `1` |
| `ENABLE_DOUBLE_RUN` | 是否先预热一次、第二次计为正式结果 | `True` |
| `MAX_RETRIES` / `BENCH_MAX_ERRORS` | 失败重试次数 / 连续失败上限 | `2` / `3` |
| `SUBPROCESS_TIMEOUT` | 单次子进程超时（秒） | `3600` |
| `PERF_LOG_DIR` | perf_log 输出目录 | `./bench/perf_log` |

`IO` 中每个用例的字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `input_len` / `output_len` | 是 | 输入 / 输出 token 长度 |
| `concurrency` | 是 | 该组合要跑的并发点列表（写单个 int 等价于 `[int]`） |
| `num_prompts_ratio` | 否 | 「请求数 : 并发数」比例，缺省用全局 `NUM_PROMPTS_RATIO` |
| `ttft_max` / `tpot_max` | 否 | 该用例的阈值 (ms)，缺省用全局默认值 |

请求数 = `ceil(并发 × 比例)`；比例 ≥ 1 时保证不少于并发数。

---

## 输出说明

运行产物统一写入当前工作目录下的 `bench/`：

```
bench/
├── best_metrics_YYYYMMDD_HHMMSS.csv          # 最优并发点的全量指标（每用例×数据集一行）
├── log/
│   └── YYYYMMDD/                             # 按运行日期归档
│       ├── summary_YYYYMMDD_HHMMSS.csv       # 汇总：最优并发 / 阈值 / 达标点数等
│       └── context_<il>x<ol>/                # 按「输入长度×输出长度」分目录
│           ├── vllm_bench_result-*.csv       # 每个并发点的全部提取指标（追加写）
│           └── sweep_results-*.csv           # 每个并发点一行：ttft / tpot / passed
└── perf_log/
    └── <模型名>/
        └── il*_ol*_np*_mc*_<dataset>.log     # 原始子进程输出 + 提取的指标
```

- `sweep_results` 的 `passed=1` 表示该点 TTFT 与 TPOT 同时在阈值内；
- `summary` 的 `points_passed / points_total` 为该组合的达标点数与总点数。

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

见 [CHANGELOG.md](CHANGELOG.md)。当前版本 **v1.0.0**。
