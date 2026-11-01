"""Offline checks that repeated benchmark runs preserve earlier results."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench_core import benchmark, runner
from bench_core.config import POINT_METRICS_HEADERS
from bench_core.sweep import PointResult, SweepResult


def make_sweep(concurrency, failed_concurrency=None):
    metrics = {
        'mean_ttft': 120.0,
        'mean_tpot': 25.0,
        'output_token_throughput': 40.0 * concurrency,
        'total_token_throughput': 80.0 * concurrency,
        'benchmark_duration': 3.0,
        'mean_e2el': 1500.0,
        'prefix_cache_hit_rate': 90.0,
        'spec_decode_accept_rate': 58.0,
        'bench_tool': 'sglang',
        'server_engine': 'sglang',
        'actual_dataset': 'generated-shared-prefix',
        'spec_accept_length': 2.74,
        'spec_rate_source': 'bench_accept_length',
    }
    points = [PointResult(
        concurrency=concurrency, num_prompts=concurrency * 2,
        ttft=120.0, tpot=25.0, metrics=metrics, passed=True,
    )]
    if failed_concurrency is not None:
        points.append(PointResult(
            concurrency=failed_concurrency, num_prompts=failed_concurrency * 2,
            ttft=-1, tpot=-1, failed=True,
        ))
    return SweepResult(
        input_len=1024, output_len=1024, dataset='prefix_repetition',
        num_prompts_ratio=2, pc_ratio=0.9, num_prefixes=1,
        ttft_max=3000, tpot_max=100, points=points,
    )


class CsvHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cwd_patch = patch.object(runner.os, 'getcwd', return_value=self.temp.name)
        cwd_patch.start()
        self.addCleanup(cwd_patch.stop)
        self.out_file = Path(self.temp.name) / 'bench' / 'import_all_perf.csv'

    def read_rows(self):
        with self.out_file.open(newline='', encoding='utf-8') as handle:
            return list(csv.reader(handle))

    def test_two_runs_append_successful_points_and_keep_one_header(self):
        runner._write_all_perf_csv([make_sweep(1, failed_concurrency=2)])
        first_rows = self.read_rows()
        self.assertEqual(first_rows[0], POINT_METRICS_HEADERS)
        self.assertEqual(len(first_rows), 2)

        runner._write_all_perf_csv([make_sweep(4, failed_concurrency=8)])
        rows = self.read_rows()
        self.assertEqual(rows[:len(first_rows)], first_rows)
        self.assertEqual(rows.count(POINT_METRICS_HEADERS), 1)
        self.assertEqual(len(rows), 3)
        records = [dict(zip(rows[0], row)) for row in rows[1:]]
        self.assertEqual([row['concurrency'] for row in records], ['1', '4'])
        self.assertTrue(all(row['spec_decode_accept_rate'] == '58.0' for row in records))
        self.assertTrue(all(row['spec_rate_source'] == 'bench_accept_length' for row in records))
        self.assertTrue(all(len(row) == len(POINT_METRICS_HEADERS) for row in rows))

    def test_existing_empty_file_gets_header_only_once(self):
        self.out_file.parent.mkdir(parents=True)
        self.out_file.touch()
        runner._write_all_perf_csv([make_sweep(1)])
        runner._write_all_perf_csv([make_sweep(2)])
        rows = self.read_rows()
        self.assertEqual(rows[0], POINT_METRICS_HEADERS)
        self.assertEqual(rows.count(POINT_METRICS_HEADERS), 1)
        self.assertEqual(len(rows), 3)

    def test_header_only_file_accepts_first_data_without_second_header(self):
        self.out_file.parent.mkdir(parents=True)
        with self.out_file.open('w', newline='', encoding='utf-8') as handle:
            csv.writer(handle).writerow(POINT_METRICS_HEADERS)
        runner._write_all_perf_csv([make_sweep(2)])
        rows = self.read_rows()
        self.assertEqual(rows[0], POINT_METRICS_HEADERS)
        self.assertEqual(rows.count(POINT_METRICS_HEADERS), 1)
        self.assertEqual(len(rows), 2)

    def test_legacy_prefix_header_preserves_old_values_and_pads_new_columns(self):
        # v1.3.4 contained these sixteen columns, before tool/source metadata.
        legacy_header = POINT_METRICS_HEADERS[:16]
        legacy_row = [f'old-{column}' for column in legacy_header]
        legacy_row[0] = '旧场景,含逗号'
        self.out_file.parent.mkdir(parents=True)
        with self.out_file.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(legacy_header)
            writer.writerow(legacy_row)

        runner._write_all_perf_csv([make_sweep(4)])
        rows = self.read_rows()
        self.assertEqual(rows[0], POINT_METRICS_HEADERS)
        self.assertEqual(rows[1][:len(legacy_header)], legacy_row)
        self.assertEqual(rows[1][len(legacy_header):],
                         [''] * (len(POINT_METRICS_HEADERS) - len(legacy_header)))
        self.assertEqual(len(rows), 3)

        runner._write_all_perf_csv([make_sweep(8)])
        appended = self.read_rows()
        self.assertEqual(appended[:len(rows)], rows)
        self.assertEqual(appended.count(POINT_METRICS_HEADERS), 1)
        self.assertEqual(len(appended), 4)

    def test_unknown_header_is_rejected_without_changing_original_bytes(self):
        self.out_file.parent.mkdir(parents=True)
        original = b'unknown,header\r\noriginal,data\r\n'
        self.out_file.write_bytes(original)
        with self.assertRaises(ValueError):
            runner._write_all_perf_csv([make_sweep(4)])
        self.assertEqual(self.out_file.read_bytes(), original)

    def test_reordered_header_is_rejected_without_changing_original_bytes(self):
        self.out_file.parent.mkdir(parents=True)
        reordered = list(POINT_METRICS_HEADERS)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.out_file.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(reordered)
            writer.writerow(['previous-result'] * len(reordered))
        original = self.out_file.read_bytes()
        with self.assertRaises(ValueError):
            runner._write_all_perf_csv([make_sweep(4)])
        self.assertEqual(self.out_file.read_bytes(), original)


class PerfLogHistoryTests(unittest.TestCase):
    def test_same_case_in_two_runs_keeps_both_raw_outputs_and_fixed_basename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = {
                'bench_tool': 'sglang',
                'server_engine': 'sglang',
                'actual_dataset': 'generated-shared-prefix',
                'spec_decode_accept_rate': 58.0,
                'spec_rate_source': 'bench_accept_length',
            }
            first_stamp = '20261101_120000_000001'
            second_stamp = '20261101_120000_000002'
            basename = 'il1024_ol1024_np4_mc2.log'
            model_dataset_dir = 'test-model_prefix_repetition_pc0.9_np1_sglang'
            with patch.object(benchmark, 'PERF_LOG_DIR', temp_dir), \
                    patch.object(benchmark, 'PERF_MODEL_NAME', 'test-model'), \
                    patch.object(benchmark, 'get_bench_runtime',
                                 side_effect=AssertionError('must not initialize runtime')):
                with patch.object(benchmark, 'SCRIPT_START_STAMP', first_stamp, create=True):
                    benchmark.save_perf_log_entry(
                        1024, 1024, 2, 4, 'prefix_repetition', metrics,
                        'first run raw output\nAccept length: 2.74\n', '_pc0.9_np1',
                    )
                first_path = Path(temp_dir) / first_stamp / model_dataset_dir / basename
                first_bytes = first_path.read_bytes()
                with patch.object(benchmark, 'SCRIPT_START_STAMP', second_stamp, create=True):
                    benchmark.save_perf_log_entry(
                        1024, 1024, 2, 4, 'prefix_repetition', metrics,
                        'second run raw output\nAccept length: 3.00\n', '_pc0.9_np1',
                    )
            second_path = Path(temp_dir) / second_stamp / model_dataset_dir / basename
            self.assertEqual(first_path.read_bytes(), first_bytes)
            self.assertIn('first run raw output', first_bytes.decode('utf-8'))
            self.assertNotIn('second run raw output', first_bytes.decode('utf-8'))
            second_text = second_path.read_text(encoding='utf-8')
            self.assertIn('second run raw output', second_text)
            self.assertNotIn('first run raw output', second_text)
            self.assertIn('spec_decode_accept_rate: 58.0', second_text)
            self.assertEqual(len(list(Path(temp_dir).rglob(basename))), 2)


if __name__ == '__main__':
    unittest.main()
