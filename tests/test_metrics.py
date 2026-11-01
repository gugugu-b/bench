"""Offline regressions for benchmark log parsing and acceptance-rate precedence."""

import math
import unittest
from unittest.mock import patch

from bench_core import config
from bench_core.metrics import (
    _extract_all_metrics,
    compute_metrics_rates,
    parse_prometheus_text,
    select_spec_accept_rate,
)


SGLANG_LOG = """
Generated shared prefix dataset statistics:
Total input tokens: 999999
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     4
Benchmark duration (s):                  259.91
Total input tokens:                      524288
Total input text tokens:                 524288
Total generated tokens (retokenized):     1922
Total generated tokens:                  4096
Request throughput (req/s):              0.02
Peak output token throughput (tok/s):     64.00
Output token throughput (tok/s):          15.76
Total token throughput (tok/s):           2032.92
Concurrency:                             1.00
Accept length:                           2.74
Mean E2E Latency (ms):                    259000.00
Mean TTFT (ms):                           1000.00
Median TTFT (ms):                         900.00
P99 TTFT (ms):                            1500.00
Mean TPOT (ms):                           50.00
Median TPOT (ms):                         49.00
P99 TPOT (ms):                            60.00
Mean ITL (ms):                            20.00
Median ITL (ms):                          19.00
P99 ITL (ms):                             30.00
"""


class BenchmarkLogTests(unittest.TestCase):
    def test_sglang_result_ignores_preparation_peak_and_retokenized_values(self):
        metrics = _extract_all_metrics(SGLANG_LOG)
        self.assertEqual(metrics['successful_requests'], 4)
        self.assertEqual(metrics['total_input_tokens'], 524288)
        self.assertEqual(metrics['total_generated_tokens'], 4096)
        self.assertEqual(metrics['output_token_throughput'], 15.76)
        self.assertEqual(metrics['benchmark_duration'], 259.91)
        self.assertEqual(metrics['mean_e2el'], 259000)
        self.assertEqual(metrics['spec_accept_length'], 2.74)
        self.assertEqual(metrics['mean_ttft'], 1000)
        self.assertEqual(metrics['mean_tpot'], 50)
        self.assertEqual(metrics['p99_itl'], 30)

    def test_last_result_table_is_used_and_ansi_is_removed(self):
        log = SGLANG_LOG + (
            '\x1b[32m============ Serving Benchmark Result ============\x1b[0m\n'
            'Successful requests: 2\nMean TTFT (ms): 12.50\n'
        )
        metrics = _extract_all_metrics(log)
        self.assertEqual(metrics['successful_requests'], 2)
        self.assertEqual(metrics['mean_ttft'], 12.5)
        self.assertTrue(math.isinf(metrics['mean_tpot']))

    def test_legacy_vllm_log_without_result_header(self):
        metrics = _extract_all_metrics(
            'Successful requests: 8\nMean E2EL (ms): 23.45\n'
            'Mean TTFT (ms): 10\nAcceptance rate (%): 67.64\n'
        )
        self.assertEqual(metrics['successful_requests'], 8)
        self.assertEqual(metrics['mean_e2el'], 23.45)
        self.assertEqual(metrics['spec_accept_rate'], 67.64)
        self.assertEqual(
            select_spec_accept_rate(metrics, 'vllm', 'vllm'),
            (67.64, 'bench_acceptance_rate'),
        )


class BenchAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.config_patch = patch.multiple(
            config,
            SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS=4,
            SGLANG_SPECULATIVE_NUM_STEPS=3,
            create=True,
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def test_user_eagle_configuration_and_log_produce_58_percent(self):
        self.assertEqual(
            select_spec_accept_rate(_extract_all_metrics(SGLANG_LOG), 'sglang', 'sglang'),
            (58.0, 'bench_accept_length'),
        )

    def test_direct_percentage_including_zero_has_priority_over_length(self):
        for rate in (0, 75.25, 100):
            with self.subTest(rate=rate):
                metrics = {'spec_accept_rate': rate, 'spec_accept_length': 2.74}
                self.assertEqual(
                    select_spec_accept_rate(metrics, 'sglang', 'sglang'),
                    (rate, 'bench_acceptance_rate'),
                )

    def test_server_configuration_has_priority_over_manual_defaults(self):
        server_info = {
            'speculative_algorithm': 'EAGLE',
            'speculative_num_draft_tokens': 5,
            'speculative_num_steps': 3,
        }
        self.assertEqual(
            select_spec_accept_rate({'spec_accept_length': 3}, 'sglang', 'sglang', server_info),
            (50.0, 'bench_accept_length'),
        )

    def test_pd_server_args_configuration_and_steps_fallback(self):
        server_info = {'decode': [{'server_args': {
            'speculative_algorithm': 'EAGLE',
            'speculative_num_draft_tokens': None,
            'speculative_num_steps': 4,
        }}]}
        self.assertEqual(
            select_spec_accept_rate({'spec_accept_length': 3}, 'sglang', 'sglang', server_info),
            (50.0, 'bench_accept_length'),
        )

    def test_manual_steps_are_used_when_draft_count_is_unset(self):
        with patch.object(config, 'SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS', None):
            self.assertEqual(
                select_spec_accept_rate({'spec_accept_length': 2.74}, 'sglang', 'sglang'),
                (58.0, 'bench_accept_length'),
            )

    def test_no_valid_conversion_without_log_or_denominator(self):
        self.assertEqual(select_spec_accept_rate({}, 'sglang', 'sglang'), ('', ''))
        with patch.multiple(
            config, SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS=None,
            SGLANG_SPECULATIVE_NUM_STEPS=None,
        ):
            self.assertEqual(
                select_spec_accept_rate({'spec_accept_length': 2.74}, 'sglang', 'sglang'),
                ('', ''),
            )

    def test_length_conversion_is_only_for_sglang_tool_and_server(self):
        for tool, server in [('vllm', 'sglang'), ('sglang', 'vllm'), ('sglang', '')]:
            with self.subTest(tool=tool, server=server):
                self.assertEqual(
                    select_spec_accept_rate({'spec_accept_length': 2.74}, tool, server),
                    ('', ''),
                )

    def test_invalid_acceptance_values_are_not_clamped(self):
        for length in [float('nan'), float('inf'), -1, 0, 0.99, 4.01, True, 'bad']:
            with self.subTest(length=length):
                self.assertEqual(
                    select_spec_accept_rate({'spec_accept_length': length}, 'sglang', 'sglang'),
                    ('', ''),
                )
        for rate in [float('nan'), float('inf'), -1, 101, True, 'bad']:
            with self.subTest(rate=rate):
                self.assertEqual(
                    select_spec_accept_rate({'spec_accept_rate': rate}, 'sglang', 'sglang'),
                    ('', ''),
                )

    def test_acceptance_length_endpoints_are_valid(self):
        for length, rate in [(1, 0), (4, 100)]:
            with self.subTest(length=length):
                self.assertEqual(
                    select_spec_accept_rate({'spec_accept_length': length}, 'sglang', 'sglang'),
                    (rate, 'bench_accept_length'),
                )

    def test_unsupported_algorithm_or_invalid_server_parameters_fall_back(self):
        configurations = [
            {'speculative_algorithm': None},
            {'speculative_algorithm': 'NGRAM'},
            {'speculative_num_draft_tokens': 1},
            {'speculative_num_draft_tokens': 'bad'},
            {'speculative_num_draft_tokens': 3.5},
            {'speculative_num_draft_tokens': None, 'speculative_num_steps': 0},
        ]
        for server_info in configurations:
            with self.subTest(server_info=server_info):
                self.assertEqual(
                    select_spec_accept_rate({'spec_accept_length': 2.74}, 'sglang', 'sglang', server_info),
                    ('', ''),
                )


class MetricsFallbackTests(unittest.TestCase):
    def test_vllm_counter_delta_and_total_suffix(self):
        before = parse_prometheus_text('''
vllm:prefix_cache_hits_total 10
vllm:prefix_cache_queries_total 20
vllm:spec_decode_num_accepted_tokens_total 30
vllm:spec_decode_num_draft_tokens_total 40
''')
        after = parse_prometheus_text('''
vllm:prefix_cache_hits_total 30
vllm:prefix_cache_queries_total 60
vllm:spec_decode_num_accepted_tokens_total 75
vllm:spec_decode_num_draft_tokens_total 100
''')
        self.assertEqual(compute_metrics_rates(before, after), (50.0, 75.0))

    def test_sglang_cache_delta_and_sampled_active_ranks(self):
        before = parse_prometheus_text('''
sglang:cached_tokens_total 100
sglang:prompt_tokens_total 200
''')
        after = parse_prometheus_text('''
sglang:cached_tokens_total 180
sglang:prompt_tokens_total 300
sglang:spec_accept_rate{dp_rank="0"} 1
sglang:spec_accept_length{dp_rank="0"} 4
''')
        sampled = parse_prometheus_text('''
sglang:spec_accept_rate{dp_rank="0"} 0.93
sglang:spec_accept_length{dp_rank="0"} 3.79
sglang:spec_accept_rate{dp_rank="1"} 0
sglang:spec_accept_length{dp_rank="1"} 0
''')
        self.assertEqual(compute_metrics_rates(before, after, [sampled]), (80.0, 93.0))

    def test_nonempty_samples_without_acceptance_metric_use_after_snapshot(self):
        after = parse_prometheus_text('sglang:spec_accept_rate 0.9\n')
        invalid_samples = [parse_prometheus_text('sglang:prompt_tokens_total 1\n')]
        self.assertEqual(compute_metrics_rates({}, after, invalid_samples), ('', 90.0))

    def test_unavailable_metrics_remain_empty(self):
        self.assertEqual(compute_metrics_rates(None, None), ('', ''))
        self.assertEqual(compute_metrics_rates({}, {'unrelated': [('', 1)]}), ('', ''))


if __name__ == '__main__':
    unittest.main()
