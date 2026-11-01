"""压测工具选择、服务端只读识别与命令构造。"""

import json
import logging
import math
import os
import re
import urllib.request
from dataclasses import dataclass, field

from . import config


@dataclass
class BenchRuntime:
    """一次运行共享的工具选择与服务端信息,不在各并发点重复识别。"""

    tool: str
    server_engine: str = ""
    selection: str = "manual"
    server_info: dict = field(default_factory=dict)


_DATASET_FAMILIES = {
    "random": "random",
    "random-ids": "random",
    "prefix_repetition": "prefix_repetition",
    "generated-shared-prefix": "prefix_repetition",
}
_METRIC_ENGINE_RE = re.compile(r"^(vllm|sglang):", re.MULTILINE)


def _tool_name(tool: str) -> str:
    name = str(tool).strip().lower()
    if name not in ("vllm", "sglang"):
        raise ValueError(f"不支持的 BENCH_TOOL: {tool!r}; 可选 vllm / sglang")
    return name


def actual_dataset(dataset: str, tool: str) -> str:
    """把逻辑名称或原生别名映射为选定工具的数据集名称。"""
    tool = _tool_name(tool)
    try:
        family = _DATASET_FAMILIES[dataset]
    except KeyError as e:
        raise ValueError(f"不支持的数据集: {dataset!r}") from e
    if tool == "vllm":
        return family
    return "random-ids" if family == "random" else "generated-shared-prefix"


def _request_text(path: str):
    """只读探测失败返回 None;不受 ENABLE_METRICS_SCRAPE 开关影响。"""
    url = f"http://{config.HOST}:{config.PORT}{path}"
    headers = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif os.environ.get("API_KEY"):
        headers["Authorization"] = os.environ["API_KEY"]
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=config.BENCH_DETECT_TIMEOUT) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _request_json(path: str):
    text = _request_text(path)
    if text is None:
        return None
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _load_sglang_server_info() -> dict:
    """读取 SGLang 信息,兼容旧路由和 PD 网关的 decode 包装。"""
    for path in ("/server_info", "/get_server_info"):
        value = _request_json(path)
        if not value:
            continue
        decode = value.get("decode")
        if isinstance(decode, list) and decode and isinstance(decode[0], dict):
            value = decode[0]
        # 单独的 version/model 字段不构成 SGLang 身份证据。
        if "internal_states" in value or (
            "model_path" in value
            and any(key in value for key in ("speculative_algorithm", "tp_size", "dp_size"))
        ):
            return value
    return {}


def resolve_bench_runtime() -> BenchRuntime:
    """启动时选择工具;未知或冲突时告警并回退手工 BENCH_TOOL。"""
    manual_tool = _tool_name(config.BENCH_TOOL)
    # SGLang 原生请求协议本身已明确指定目标类型。即使信息接口/metrics 不开放,
    # 仍可用 bench Accept length 和手工投机参数换算,不影响默认 vLLM 的识别行为。
    manual_engine = (
        "sglang" if manual_tool == "sglang" and config.SGLANG_BACKEND.startswith("sglang") else ""
    )
    if not config.AUTO_SELECT_BENCH:
        info = _load_sglang_server_info() if manual_tool == "sglang" else {}
        return BenchRuntime(manual_tool, "sglang" if info else manual_engine, "manual", info)

    info = _load_sglang_server_info()
    metrics_text = _request_text(config.METRICS_SCRAPE_PATH) or ""
    candidates = set(_METRIC_ENGINE_RE.findall(metrics_text))
    if info:
        candidates.add("sglang")
    version = _request_json("/version")
    if version and isinstance(version.get("version"), str):
        candidates.add("vllm")
    if len(candidates) == 1:
        engine = candidates.pop()
        logging.info(f"自动识别服务端: {engine}; 选择 {engine} bench")
        return BenchRuntime(engine, engine, "auto", info)

    reason = "服务端特征冲突" if candidates else "没有取得可识别的服务端特征"
    logging.warning(f"{reason}; 回退手工配置 BENCH_TOOL={manual_tool}")
    return BenchRuntime(manual_tool, manual_engine, "fallback", info)


def build_bench_command(runtime: BenchRuntime, input_len: int, output_len: int,
                        concurrency: int, dataset: str, num_prompts: int,
                        pc_ratio: float, num_prefixes: int) -> list:
    """生成命令;SGLang 内置预热关闭,由项目统一执行外部预热轮。"""
    tool = _tool_name(runtime.tool)
    selected_dataset = actual_dataset(dataset, tool)
    prefix_dataset = selected_dataset in ("prefix_repetition", "generated-shared-prefix")
    if prefix_dataset:
        if not math.isfinite(pc_ratio) or not 0 <= pc_ratio <= 1:
            raise ValueError(f"pc_ratio 必须在 [0, 1] 内: {pc_ratio}")
        if num_prefixes <= 0:
            raise ValueError(f"num_prefixes 必须大于 0: {num_prefixes}")
        # 与既有 vLLM 命令保持相同的两段取整规则。
        prefix_len = math.ceil(input_len * pc_ratio)
        suffix_len = math.ceil(input_len * (1 - pc_ratio))

    if tool == "vllm":
        command = [
            "vllm", "bench", "serve",
            "--host", config.HOST,
            "--port", str(config.PORT),
            "--backend", config.BACKEND,
            "--served-model-name", config.SERVED_MODEL_NAME,
            "--model", config.MODEL,
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(concurrency),
            "--temperature", str(config.TEMPERATURE),
            "--percentile-metrics", config.PERCENTILE_METRICS,
        ]
        if prefix_dataset:
            command += [
                "--dataset-name", selected_dataset,
                "--prefix-repetition-prefix-len", str(prefix_len),
                "--prefix-repetition-suffix-len", str(suffix_len),
                "--prefix-repetition-output-len", str(output_len),
                "--prefix-repetition-num-prefixes", str(num_prefixes),
            ]
        else:
            command += [
                "--dataset-name", selected_dataset,
                "--random-input-len", str(input_len),
                "--random-output-len", str(output_len),
            ]
        if config.IGNORE_EOS:
            command.append(config.IGNORE_EOS)
        return command

    command = [
        config.SGLANG_PYTHON, "-m", config.SGLANG_BENCH_MODULE,
        "--host", config.HOST,
        "--port", str(config.PORT),
        "--backend", config.SGLANG_BACKEND,
        "--served-model-name", config.SERVED_MODEL_NAME,
        "--model", config.MODEL,
        "--tokenizer", config.MODEL,
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--warmup-requests", "0",
        "--temperature", str(config.TEMPERATURE),
        # 已由项目保存 stdout 与 CSV,避免 SGLang 在 cwd 另建默认 JSONL。
        "--output-file", os.devnull,
        "--dataset-name", selected_dataset,
    ]
    if not config.IGNORE_EOS:
        command.append("--disable-ignore-eos")
    if config.SGLANG_BACKEND not in ("sglang", "sglang-native"):
        # v0.5.18 的 OpenAI 请求适配器从 extra_request_body 读取 temperature。
        command += ["--extra-request-body", json.dumps({"temperature": config.TEMPERATURE})]
    if prefix_dataset:
        if num_prompts <= 0 or num_prompts % num_prefixes:
            raise ValueError(
                f"generated-shared-prefix 请求数 {num_prompts} 必须能被前缀组数 "
                f"num_prefixes={num_prefixes} 整除"
            )
        command += [
            "--gsp-num-groups", str(num_prefixes),
            "--gsp-prompts-per-group", str(num_prompts // num_prefixes),
            "--gsp-system-prompt-len", str(prefix_len),
            "--gsp-question-len", str(suffix_len),
            "--gsp-output-len", str(output_len),
            "--gsp-range-ratio", "1",
        ]
    else:
        command += [
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            "--random-range-ratio", "1",
        ]
    return command
