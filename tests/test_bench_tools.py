import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from bench_core import config
from bench_core.bench_tools import (
    VLLM_BENCH_DIRECT_SCRIPT,
    BenchRuntime,
    actual_dataset,
    build_bench_command,
    can_fallback_vllm_cli,
    resolve_bench_runtime,
    should_fallback_vllm_cli,
    summarize_cli_error,
    to_vllm_direct_command,
)
from bench_core import bench_tools


class BenchToolsTest(unittest.TestCase):
    def setUp(self):
        overrides = {
            "BENCH_TOOL": "vllm", "AUTO_SELECT_BENCH": False,
            "BENCH_DETECT_TIMEOUT": 3, "VLLM_PYTHON": sys.executable,
            "VLLM_BENCH_MODE": "auto", "SGLANG_PYTHON": sys.executable,
            "SGLANG_BENCH_MODULE": "sglang.bench_serving", "SGLANG_BACKEND": "sglang",
            "HOST": "127.0.0.1", "PORT": "30000", "BACKEND": "vllm",
            "SERVED_MODEL_NAME": "served-name", "MODEL": "/model/tokenizer",
            "TEMPERATURE": 0, "PERCENTILE_METRICS": "ttft,tpot,itl,e2el",
            "IGNORE_EOS": "--ignore-eos", "METRICS_SCRAPE_PATH": "/metrics",
        }
        settings = patch.multiple(config, create=True, **overrides)
        settings.start()
        self.addCleanup(settings.stop)

    def build(self, tool="vllm", dataset="random", num_prompts=12,
              pc_ratio=0.25, num_prefixes=1):
        return build_bench_command(
            BenchRuntime(tool), 1024, 128, 4, dataset, num_prompts, pc_ratio, num_prefixes,
        )

    @staticmethod
    def value(command, flag):
        return command[command.index(flag) + 1]

    def test_dataset_aliases_map_in_both_directions(self):
        for alias in ("random", "random-ids"):
            self.assertEqual(actual_dataset(alias, "vllm"), "random")
            self.assertEqual(actual_dataset(alias, "sglang"), "random-ids")
        for alias in ("prefix_repetition", "generated-shared-prefix"):
            self.assertEqual(actual_dataset(alias, "vllm"), "prefix_repetition")
            self.assertEqual(actual_dataset(alias, "sglang"), "generated-shared-prefix")
        with self.assertRaises(ValueError):
            actual_dataset("typo", "sglang")
        with self.assertRaises(ValueError):
            actual_dataset("random", "typo")

    def test_vllm_default_command_keeps_existing_flags(self):
        self.assertEqual(self.build(), [
            "vllm", "bench", "serve", "--host", "127.0.0.1", "--port", "30000",
            "--backend", "vllm", "--served-model-name", "served-name",
            "--model", "/model/tokenizer", "--num-prompts", "12",
            "--max-concurrency", "4", "--temperature", "0",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--dataset-name", "random",
            "--random-input-len", "1024", "--random-output-len", "128", "--ignore-eos",
        ])

    def test_prefix_mapping_preserves_existing_lengths(self):
        command = self.build(dataset="generated-shared-prefix", pc_ratio=0.1)
        self.assertEqual(self.value(command, "--dataset-name"), "prefix_repetition")
        self.assertEqual(self.value(command, "--prefix-repetition-prefix-len"), "103")
        self.assertEqual(self.value(command, "--prefix-repetition-suffix-len"), "922")

    def test_sglang_random_fixed_lengths_and_external_warmup(self):
        command = self.build("sglang")
        self.assertEqual(command[:3], [sys.executable, "-m", "sglang.bench_serving"])
        self.assertEqual(self.value(command, "--dataset-name"), "random-ids")
        self.assertEqual(self.value(command, "--random-input-len"), "1024")
        self.assertEqual(self.value(command, "--random-output-len"), "128")
        self.assertEqual(self.value(command, "--random-range-ratio"), "1")
        self.assertEqual(self.value(command, "--warmup-requests"), "0")
        self.assertEqual(self.value(command, "--request-rate"), "inf")
        self.assertEqual(self.value(command, "--tokenizer"), "/model/tokenizer")
        self.assertEqual(self.value(command, "--output-file"), os.devnull)
        self.assertNotIn("--percentile-metrics", command)
        self.assertNotIn("--tokenize-prompt", command)
        self.assertNotIn("--flush-cache", command)
        self.assertEqual(command, self.build("sglang", "random-ids"))

    def test_sglang_gsp_maps_prefix_and_request_counts(self):
        command = self.build("sglang", "prefix_repetition")
        self.assertEqual(self.value(command, "--dataset-name"), "generated-shared-prefix")
        self.assertEqual(self.value(command, "--gsp-num-groups"), "1")
        self.assertEqual(self.value(command, "--gsp-prompts-per-group"), "12")
        self.assertEqual(self.value(command, "--gsp-system-prompt-len"), "256")
        self.assertEqual(self.value(command, "--gsp-question-len"), "768")
        self.assertEqual(self.value(command, "--gsp-output-len"), "128")
        self.assertEqual(self.value(command, "--gsp-range-ratio"), "1")
        self.assertEqual(command, self.build("sglang", "generated-shared-prefix"))

    def test_sglang_gsp_does_not_silently_change_total_requests(self):
        with self.assertRaisesRegex(ValueError, "请求数 12.*num_prefixes=5"):
            self.build("sglang", "prefix_repetition", num_prefixes=5)
        command = self.build("sglang", "prefix_repetition", num_prefixes=3)
        self.assertEqual(self.value(command, "--gsp-prompts-per-group"), "4")

    def test_sglang_respects_eos_setting(self):
        self.assertNotIn("--disable-ignore-eos", self.build("sglang"))
        with patch.object(config, "IGNORE_EOS", ""):
            self.assertIn("--disable-ignore-eos", self.build("sglang"))
            self.assertNotIn("", self.build())

    def test_sglang_openai_adapter_receives_temperature(self):
        with patch.multiple(config, SGLANG_BACKEND="vllm", TEMPERATURE=0.7):
            command = self.build("sglang")
        extra = json.loads(self.value(command, "--extra-request-body"))
        self.assertEqual(extra, {"temperature": 0.7})

    def test_invalid_prefix_configuration_fails_early(self):
        for tool in ("vllm", "sglang"):
            for ratio in (-0.1, 1.1, float("nan")):
                with self.subTest(tool=tool, ratio=ratio), self.assertRaises(ValueError):
                    self.build(tool, "prefix_repetition", pc_ratio=ratio)
            with self.assertRaises(ValueError):
                self.build(tool, "prefix_repetition", num_prefixes=0)

    def test_manual_default_does_not_probe_or_switch_tools(self):
        with patch.object(bench_tools, "_request_text") as request:
            runtime = resolve_bench_runtime()
        request.assert_not_called()
        self.assertEqual(runtime, BenchRuntime("vllm"))

    def test_manual_sglang_reads_server_configuration(self):
        info = {"model_path": "/model", "internal_states": [], "speculative_num_draft_tokens": 4}
        with patch.object(config, "BENCH_TOOL", "sglang"), patch.object(
            bench_tools, "_request_text", return_value=json.dumps(info),
        ):
            runtime = resolve_bench_runtime()
        self.assertEqual(runtime.tool, "sglang")
        self.assertEqual(runtime.selection, "manual")
        self.assertEqual(runtime.server_engine, "sglang")
        self.assertEqual(runtime.server_info, info)

    def test_legacy_and_pd_server_info_are_normalized(self):
        info = {"internal_states": [], "speculative_num_draft_tokens": 4}
        replies = {"/get_server_info": json.dumps({"decode": [info]})}
        with patch.object(config, "BENCH_TOOL", "sglang"), patch.object(
            bench_tools, "_request_text", side_effect=lambda path: replies.get(path),
        ) as request:
            runtime = resolve_bench_runtime()
        self.assertEqual(runtime.server_info, info)
        self.assertEqual(request.call_count, 2)

    def test_manual_sglang_server_info_failure_is_optional(self):
        with patch.object(config, "BENCH_TOOL", "sglang"), patch.object(
            bench_tools, "_request_text", return_value=None,
        ):
            runtime = resolve_bench_runtime()
        self.assertEqual(runtime.tool, "sglang")
        self.assertEqual(runtime.server_info, {})

    def auto_runtime(self, responses, **settings):
        with patch.multiple(config, AUTO_SELECT_BENCH=True, **settings), patch.object(
            bench_tools, "_request_text", side_effect=lambda path: responses.get(path),
        ):
            return resolve_bench_runtime()

    def test_auto_detects_sglang_from_native_info(self):
        info = {"model_path": "/model", "internal_states": []}
        runtime = self.auto_runtime({"/server_info": json.dumps(info)})
        self.assertEqual(runtime.tool, "sglang")
        self.assertEqual(runtime.server_engine, "sglang")
        self.assertEqual(runtime.selection, "auto")

    def test_auto_detects_metrics_without_metrics_scrape_enabled(self):
        for engine in ("vllm", "sglang"):
            with self.subTest(engine=engine):
                runtime = self.auto_runtime({"/metrics": f"{engine}:request_count 4\n"}, ENABLE_METRICS_SCRAPE=False)
                self.assertEqual(runtime.tool, engine)
                self.assertEqual(runtime.selection, "auto")

    def test_auto_detects_vllm_version_endpoint(self):
        runtime = self.auto_runtime({"/version": '{"version":"0.18.0"}'})
        self.assertEqual(runtime.tool, "vllm")
        self.assertEqual(runtime.selection, "auto")

    def test_unknown_and_conflicting_servers_fall_back(self):
        for responses in ({}, {"/metrics": "vllm:a 1\nsglang:b 1\n"},
                          {"/server_info": '{"internal_states":[]}', "/version": '{"version":"1"}'}):
            for manual_tool in ("vllm", "sglang"):
                with self.subTest(responses=responses, manual_tool=manual_tool):
                    with self.assertLogs(level="WARNING"):
                        runtime = self.auto_runtime(responses, BENCH_TOOL=manual_tool)
                    self.assertEqual(runtime.tool, manual_tool)
                    self.assertEqual(runtime.selection, "fallback")

    def test_generic_or_malformed_json_is_not_sglang_evidence(self):
        for response in ('{"version":"1","model":"foo"}', "<html>not found</html>", "[]"):
            with self.assertLogs(level="WARNING"):
                runtime = self.auto_runtime({"/server_info": response})
            self.assertEqual(runtime.selection, "fallback")

    def test_probe_passes_timeout_and_network_failure_is_optional(self):
        with patch.object(bench_tools.urllib.request, "urlopen", side_effect=TimeoutError) as request:
            self.assertIsNone(bench_tools._request_text("/metrics"))
        self.assertEqual(request.call_args.kwargs["timeout"], 3)
        self.assertEqual(request.call_args.args[0].full_url, "http://127.0.0.1:30000/metrics")

    def test_vllm_direct_mode_builds_python_command_with_identical_flags(self):
        for dataset in ("random", "prefix_repetition"):
            with self.subTest(dataset=dataset):
                cli_cmd = build_bench_command(
                    BenchRuntime("vllm", vllm_launcher="cli"),
                    1024, 128, 4, dataset, 12, 0.25, 1,
                )
                direct_cmd = build_bench_command(
                    BenchRuntime("vllm", vllm_launcher="direct"),
                    1024, 128, 4, dataset, 12, 0.25, 1,
                )
                self.assertEqual(direct_cmd[:3], [sys.executable, "-c", VLLM_BENCH_DIRECT_SCRIPT])
                self.assertEqual(direct_cmd[3:], cli_cmd[3:])
                self.assertEqual(to_vllm_direct_command(cli_cmd), direct_cmd)

        with patch.object(config, "VLLM_PYTHON", "/opt/vllm/bin/python"):
            custom_cmd = build_bench_command(
                BenchRuntime("vllm", vllm_launcher="direct"),
                1024, 128, 4, "random", 12, 0.25, 1,
            )
        self.assertEqual(custom_cmd[0], "/opt/vllm/bin/python")
        with patch.object(config, "VLLM_PYTHON", ""):
            fallback_py_cmd = build_bench_command(
                BenchRuntime("vllm", vllm_launcher="direct"),
                1024, 128, 4, "random", 12, 0.25, 1,
            )
        self.assertEqual(fallback_py_cmd[0], sys.executable)

    def test_resolve_runtime_respects_vllm_bench_mode(self):
        with patch.object(config, "VLLM_BENCH_MODE", "direct"):
            runtime = resolve_bench_runtime()
        self.assertEqual(runtime.tool, "vllm")
        self.assertEqual(runtime.vllm_launcher, "direct")

        with patch.object(config, "VLLM_BENCH_MODE", "direct"):
            auto_vllm = self.auto_runtime({"/version": '{"version":"0.18.0"}'})
        self.assertEqual(auto_vllm.tool, "vllm")
        self.assertEqual(auto_vllm.vllm_launcher, "direct")

        with patch.multiple(config, BENCH_TOOL="sglang", VLLM_BENCH_MODE="direct"), patch.object(
            bench_tools, "_request_text", return_value=None,
        ):
            sglang_rt = resolve_bench_runtime()
        self.assertEqual(sglang_rt.tool, "sglang")
        self.assertEqual(sglang_rt.vllm_launcher, "cli")

        with patch.object(config, "VLLM_BENCH_MODE", "typo"), self.assertRaises(ValueError):
            resolve_bench_runtime()

    def test_to_vllm_direct_command_rejects_non_cli_commands(self):
        for cmd in ([sys.executable, "-m", "sglang.bench_serving"],
                    [sys.executable, "-c", VLLM_BENCH_DIRECT_SCRIPT, "--host", "127.0.0.1"]):
            with self.assertRaises(ValueError):
                to_vllm_direct_command(cmd)

    def test_should_fallback_vllm_cli_detects_startup_import_and_missing_binary(self):
        cli_cmd = ["vllm", "bench", "serve", "--host", "127.0.0.1"]
        namespace_err = subprocess.CalledProcessError(
            1, cli_cmd, output=(
                "Traceback (most recent call last):\n"
                '  File "/usr/local/bin/vllm", line 5, in <module>\n'
                "    from vllm.entrypoints.cli.main import main\n"
                "ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses' "
                "(/usr/local/lib/python3.10/dist-packages/openai/types/responses/__init__.py)\n"
            ),
        )
        module_err = subprocess.CalledProcessError(
            1, cli_cmd, output="ModuleNotFoundError: No module named 'openai.types.responses'\n",
        )
        ansi_err = subprocess.CalledProcessError(
            1, cli_cmd,
            output="\x1b[1;35mImportError\x1b[0m: cannot import name 'NamespaceTool'\n",
        )
        entrypoint_attr_err = subprocess.CalledProcessError(
            1, cli_cmd, output=(
                "Traceback (most recent call last):\n"
                '  File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/protocol.py", line 18, in <module>\n'
                "AttributeError: module 'openai.types.responses' has no attribute 'NamespaceTool'\n"
            ),
        )
        missing_bin = FileNotFoundError(2, "No such file or directory", "vllm")

        for err in (namespace_err, module_err, ansi_err, entrypoint_attr_err, missing_bin):
            with self.subTest(err=repr(err)):
                self.assertTrue(should_fallback_vllm_cli(err))
        self.assertIn("NamespaceTool", summarize_cli_error(namespace_err))

        runtime_connect_err = subprocess.CalledProcessError(
            1, cli_cmd, output=(
                "Traceback (most recent call last):\n"
                '  File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 35, in cmd\n'
                '  File "/usr/local/lib/python3.10/dist-packages/vllm/benchmarks/serve.py", line 910, in main\n'
                "aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host 127.0.0.1:30000\n"
            ),
        )
        bad_arg_err = subprocess.CalledProcessError(
            2, cli_cmd, output="vllm bench serve: error: unrecognized arguments: --unknown\n",
        )
        post_result_err = subprocess.CalledProcessError(
            1, cli_cmd, output="============ Serving Benchmark Result ============\nImportError: cleanup\n",
        )
        timeout_err = subprocess.TimeoutExpired(cli_cmd, 10)

        for err in (runtime_connect_err, bad_arg_err, post_result_err, timeout_err):
            with self.subTest(err=repr(err)):
                self.assertFalse(should_fallback_vllm_cli(err))

    def test_can_fallback_vllm_cli_checks_runtime_mode_and_command(self):
        cli_cmd = ["vllm", "bench", "serve", "--host", "127.0.0.1"]
        direct_cmd = to_vllm_direct_command(cli_cmd)
        err = subprocess.CalledProcessError(
            1, cli_cmd, output="ImportError: cannot import name 'NamespaceTool'\n",
        )
        self.assertTrue(can_fallback_vllm_cli(BenchRuntime("vllm"), cli_cmd, err))
        self.assertFalse(can_fallback_vllm_cli(BenchRuntime("vllm", vllm_launcher="direct"), cli_cmd, err))
        self.assertFalse(can_fallback_vllm_cli(BenchRuntime("vllm"), direct_cmd, err))
        self.assertFalse(can_fallback_vllm_cli(BenchRuntime("sglang"), cli_cmd, err))
        with patch.object(config, "VLLM_BENCH_MODE", "cli"):
            self.assertFalse(can_fallback_vllm_cli(BenchRuntime("vllm"), cli_cmd, err))


if __name__ == "__main__":
    unittest.main()
