"""离线验证工具选择、预热、接受率回退和最终文件之间的完整链路。"""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench_core import benchmark, config, runner
from bench_core.bench_tools import BenchRuntime
from bench_core.metrics import parse_prometheus_text


def bench_output(cmd, accept_line="Accept length: 2.74"):
    count = int(cmd[cmd.index('--num-prompts') + 1])
    concurrency = int(cmd[cmd.index('--max-concurrency') + 1])
    return f"""Total input tokens: 999999
============ Serving Benchmark Result ============
Successful requests: {count}
Benchmark duration (s): 10
Total input tokens: {count * 16}
Total generated tokens (retokenized): 1
Total generated tokens: {count * 8}
Request throughput (req/s): 0.4
Peak output token throughput (tok/s): 999
Output token throughput (tok/s): 15.76
Total token throughput (tok/s): 2032.92
Mean E2E Latency (ms): 500
Mean TTFT (ms): {concurrency * 10}
Median TTFT (ms): 10
P99 TTFT (ms): 20
Mean TPOT (ms): 4
Median TPOT (ms): 4
P99 TPOT (ms): 5
Mean ITL (ms): 4
Median ITL (ms): 4
P99 ITL (ms): 5
{accept_line}
"""


class BenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target, value in (
            ('os.getcwd', lambda: str(self.root)),
            ('bench_core.benchmark.PERF_LOG_DIR', str(self.root / 'bench' / 'perf_log')),
            ('bench_core.benchmark._bench_runtime', None),
            ('bench_core.benchmark.ENABLE_DOUBLE_RUN', True),
            ('bench_core.runner.ENABLE_DOUBLE_RUN', True),
            ('bench_core.benchmark.POST_TEST_SLEEP', 0),
            ('bench_core.csv_io.SCRIPT_START_STAMP', 'offline'),
            ('bench_core.runner.SCRIPT_START_STAMP', 'offline'),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        benchmark.reset_bench_error_counter()

    def read_csv(self, path):
        with path.open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    def formal(self, runtime, log, snapshots=(None, None)):
        benchmark.initialize_bench_runtime(runtime)
        with patch.object(benchmark, '_run_subprocess', return_value=log), \
             patch.object(benchmark, '_scrape_metrics_snapshot', side_effect=snapshots), \
             patch.object(benchmark, '_start_spec_rate_sampler', return_value={}), \
             patch.object(benchmark, '_stop_spec_rate_sampler', return_value=[]):
            return benchmark._formal_test_with_scrape(
                ['python', '-m', 'sglang.bench_serving'], 16, 8, 1, 2, 'random',
                'result.csv', 'sweep.csv', 3000, 100, '',
            )

    def test_sglang_run_warmup_is_excluded_and_all_exports_agree(self):
        runtime = BenchRuntime('sglang', 'sglang', 'auto', {
            'speculative_algorithm': 'EAGLE', 'speculative_num_draft_tokens': 4,
        })
        cases = [{
            'input_len': 16, 'output_len': 8, 'concurrency': [1, 2, 2],
            'num_prompts_ratio': 2, 'datasets': ['random', 'generated-shared-prefix'],
            'warmup_rounds': 1,
        }]
        commands = []

        def execute(cmd):
            commands.append(cmd)
            # 每点首轮预热故意用100%,正式轮才是58%。
            length = '4.00' if len(commands) % 2 else '2.74'
            return bench_output(cmd, f'Accept length: {length}')

        with patch.object(runner, 'IO', cases), \
             patch.object(benchmark, 'resolve_bench_runtime', return_value=runtime) as resolve, \
             patch.object(benchmark, '_scrape_metrics_snapshot', return_value=None), \
             patch.object(benchmark, '_run_subprocess', side_effect=execute):
            runner.run_test_cases()
        resolve.assert_called_once_with()
        self.assertEqual(len(commands), 8)  # 2数据集 × 2并发点 × 预热/正式
        for cmd in commands:
            self.assertEqual(cmd[cmd.index('--warmup-requests') + 1], '0')
        rows = self.read_csv(self.root / 'bench' / 'import_all_perf.csv')
        self.assertEqual(len(rows), 4)
        self.assertEqual({r['actual_dataset'] for r in rows},
                         {'random-ids', 'generated-shared-prefix'})
        for row in rows:
            self.assertNotIn(None, row)
            self.assertEqual(row['spec_decode_accept_rate'], '58.0')
            self.assertEqual(row['spec_accept_length'], '2.74')
            self.assertEqual(row['spec_rate_source'], 'bench_accept_length')
            self.assertEqual(row['bench_tool'], 'sglang')
            self.assertEqual(row['mean_e2el'], '500.0')
            self.assertEqual(row['output_token_throughput'], '15.76')
        summary = self.read_csv(self.root / 'bench' / 'log' / 'offline' / 'summary_offline.csv')
        best = self.read_csv(self.root / 'bench' / 'best_metrics_offline.csv')
        self.assertEqual(len(summary), 2)
        self.assertEqual([r['best_concurrency'] for r in summary], ['2', '2'])
        self.assertEqual([r['concurrency'] for r in best], ['2', '2'])
        for path in self.root.rglob('*.csv'):
            self.assertTrue(all(None not in row for row in self.read_csv(path)), path)
        logs = list((self.root / 'bench' / 'perf_log').rglob('*.log'))
        self.assertEqual(len(logs), 4)
        for log in logs:
            self.assertTrue(log.parent.name.endswith('_sglang'))
            text = log.read_text()
            self.assertIn('spec_decode_accept_rate: 58.0', text)
            self.assertIn('spec_rate_source: bench_accept_length', text)
            self.assertIn('Command:', text)
            self.assertNotIn('_raw_output', text)

    def test_log_length_works_without_info_or_metrics(self):
        # 复现手动SGLang、信息接口不开放、metrics关闭时的配置兜底。
        with patch.object(config, 'BENCH_TOOL', 'sglang'), \
             patch.object(config, 'AUTO_SELECT_BENCH', False), \
             patch('bench_core.bench_tools._request_text', return_value=None):
            runtime = benchmark.initialize_bench_runtime()
        cmd = ['--num-prompts', '2', '--max-concurrency', '1']
        _, _, metrics = self.formal(runtime, bench_output(cmd))
        self.assertEqual(metrics['spec_decode_accept_rate'], 58.0)
        self.assertEqual(metrics['spec_rate_source'], 'bench_accept_length')

    def test_missing_log_acceptance_falls_back_to_metrics_and_is_saved(self):
        before = parse_prometheus_text('sglang:spec_accept_rate{dp_rank="0"} 0.8')
        after = parse_prometheus_text('sglang:spec_accept_rate{dp_rank="0"} 0.93')
        cmd = ['--num-prompts', '2', '--max-concurrency', '1']
        _, _, metrics = self.formal(
            BenchRuntime('sglang', 'sglang'), bench_output(cmd, ''), (before, after),
        )
        self.assertEqual(metrics['spec_decode_accept_rate'], 93.0)
        self.assertEqual(metrics['spec_rate_source'], 'metrics')
        log = next(self.root.rglob('*.log')).read_text()
        self.assertIn('spec_decode_accept_rate: 93.0', log)
        self.assertIn('spec_rate_source: metrics', log)

    def test_log_zero_percent_beats_metrics(self):
        before = parse_prometheus_text('sglang:spec_accept_rate 0.8')
        after = parse_prometheus_text('sglang:spec_accept_rate 0.99')
        cmd = ['--num-prompts', '2', '--max-concurrency', '1']
        _, _, metrics = self.formal(
            BenchRuntime('vllm', 'sglang'), bench_output(cmd, 'Acceptance rate (%): 0'),
            (before, after),
        )
        self.assertEqual(metrics['spec_decode_accept_rate'], 0.0)
        self.assertEqual(metrics['spec_rate_source'], 'bench_acceptance_rate')
        self.assertFalse(next(self.root.rglob('*.log')).parent.name.endswith('_sglang'))

    def test_log_length_beats_metrics(self):
        before = parse_prometheus_text('sglang:spec_accept_rate 0.8')
        after = parse_prometheus_text('sglang:spec_accept_rate 0.99')
        cmd = ['--num-prompts', '2', '--max-concurrency', '1']
        _, _, metrics = self.formal(
            BenchRuntime('sglang', 'sglang'), bench_output(cmd), (before, after),
        )
        self.assertEqual(metrics['spec_decode_accept_rate'], 58.0)
        self.assertEqual(metrics['spec_rate_source'], 'bench_accept_length')

    def test_default_vllm_run_without_warmup_preserves_unmet_points(self):
        cases = [{
            'input_len': 16, 'output_len': 8, 'concurrency': [1],
            'datasets': ['random'], 'ttft_max': 1,
        }]
        with patch.object(runner, 'IO', cases), \
             patch.object(benchmark, 'ENABLE_DOUBLE_RUN', False), \
             patch.object(benchmark, 'ENABLE_METRICS_SCRAPE', False), \
             patch.object(benchmark, 'resolve_bench_runtime', return_value=BenchRuntime('vllm')), \
             patch.object(benchmark, '_run_subprocess',
                          side_effect=lambda cmd: bench_output(cmd, 'Acceptance rate (%): 67.64')) as run, \
             self.assertLogs(level='WARNING'):
            runner.run_test_cases()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][:3], ['vllm', 'bench', 'serve'])
        rows = self.read_csv(self.root / 'bench' / 'import_all_perf.csv')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['spec_decode_accept_rate'], '67.64')
        self.assertEqual(rows[0]['actual_dataset'], 'random')
        summary = self.read_csv(self.root / 'bench' / 'log' / 'offline' / 'summary_offline.csv')
        self.assertEqual(summary[0]['best_concurrency'], '0')
        self.assertEqual(summary[0]['points_total'], '1')
        self.assertEqual(summary[0]['points_passed'], '0')
        self.assertEqual(summary[0]['bench_tool'], 'vllm')
        self.assertTrue(list(self.root.rglob('vllm_bench_result-*.csv')))
        self.assertFalse(next(self.root.rglob('*.log')).parent.name.endswith('_sglang'))

    def test_both_sources_unavailable_are_blank(self):
        cmd = ['--num-prompts', '2', '--max-concurrency', '1']
        _, _, metrics = self.formal(BenchRuntime('vllm'), bench_output(cmd, ''))
        self.assertEqual(metrics['spec_decode_accept_rate'], '')
        self.assertEqual(metrics['spec_rate_source'], '')
        self.assertEqual(metrics['spec_accept_length'], '')

    def test_invalid_dataset_fails_before_any_benchmark(self):
        with patch.object(runner, 'IO', [{
            'input_len': 16, 'output_len': 8, 'concurrency': [1], 'datasets': ['typo'],
        }]), patch.object(benchmark, 'resolve_bench_runtime', return_value=BenchRuntime('vllm')), \
             patch.object(benchmark, '_run_subprocess') as run:
            with self.assertRaisesRegex(ValueError, '数据集'):
                runner.run_test_cases()
            run.assert_not_called()

    def test_native_dataset_alias_keeps_prefix_fields_and_warmup_settings(self):
        self.assertEqual(config.dataset_prefix_fields('generated-shared-prefix', 0.1, 1), (0.1, 1))
        self.assertEqual(config.prefix_context_tag('generated-shared-prefix', 0.1, 1), '_pc0.1_np1')
        self.assertEqual(config.dataset_prefix_fields('random-ids', 0.1, 1), ('', ''))
        self.assertEqual(config.resolve_warmup_rounds({'prefix_repetition': 4},
                                                    'generated-shared-prefix'), 4)
        self.assertEqual(config.resolve_warmup_rounds({'random-ids': 0}, 'random'), 0)


if __name__ == '__main__':
    unittest.main()
