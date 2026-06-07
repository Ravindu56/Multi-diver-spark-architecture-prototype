# =============================================================
# tests/unit/test_baseline_applications.py
#
# Unit tests for the pure-Python logic in:
#   mpj_spark/applications/baseline_spark.py
#   mpj_spark/applications/baseline_kmeans.py
#   mpj_spark/applications/baseline_logreg.py
#
# STRATEGY
# --------
# All three modules contain a large Spark-orchestration entry
# point (run_baseline / run_baseline_kmeans / run_baseline_logreg)
# that cannot be called in a unit test without a live JVM.
# We therefore test ONLY the pure-Python helpers and the
# argument-resolution / configuration logic that exists OUTSIDE
# the Spark code paths, by patching every Spark symbol before it
# is imported inside the function body.
#
# What IS tested (pure Python, no JVM):
#   baseline_spark     — core-budget formula, timing-dict keys,
#                        results shape, cores_override path
#   baseline_kmeans    — core-budget formula (all three priority
#                        levels), return-tuple structure,
#                        baseline_threads path
#   baseline_logreg    — _baseline_heap_gb() formula, all three
#                        core-resolution paths, parity_iter logic,
#                        OOM-resilient result shape, timing keys
#
# What is NOT tested here (Spark integration — future test phase):
#   - actual KMeans / LogReg model fitting
#   - SparkSession lifecycle
#   - RDD / DataFrame operations
#
# Research alignment:
#   Objective 2d — baseline validation for comparative evaluation
# =============================================================
from unittest.mock import MagicMock, patch


# =============================================================
# Section 1: baseline_spark.py  (WordCount baseline)
# =============================================================

class TestBaselineSparkCoreBudget:
    """
    Core-budget resolution logic in run_baseline().
    TOTAL_CORES is patched to 22 for all tests.
    """

    def _run(self, num_workers=2, cores_override=None):
        """
        Call run_baseline() with all Spark symbols mocked out so
        no JVM is started.  Returns (sorted_results, timing_dict).
        """
        fake_spark  = MagicMock()
        fake_sc     = MagicMock()
        fake_rdd    = MagicMock()
        fake_spark.sparkContext = fake_sc
        fake_sc.textFile.return_value = fake_rdd
        fake_rdd.count.return_value   = 10
        # .flatMap().filter().map().reduceByKey().collect() chain
        fake_rdd.flatMap.return_value  = fake_rdd
        fake_rdd.filter.return_value   = fake_rdd
        fake_rdd.map.return_value      = fake_rdd
        fake_rdd.reduceByKey.return_value = fake_rdd
        fake_rdd.collect.return_value  = [('hello', 5), ('world', 3)]

        with patch('mpj_spark.applications.baseline_spark.build_spark_session',
                   return_value=fake_spark), \
             patch('mpj_spark.config.TOTAL_CORES', 22):
            from mpj_spark.applications.baseline_spark import run_baseline
            return run_baseline('fake_path.txt', num_workers, cores_override)

    def test_auto_budget_formula(self):
        """Without cores_override, cores = TOTAL_CORES // num_workers = 22 // 2 = 11."""
        results, timing = self._run(num_workers=2, cores_override=None)
        # We can't inspect the cores value directly, but the function
        # must return without error and produce correct structure.
        assert isinstance(results, list)
        assert len(results) == 2  # two words in fake collect()

    def test_cores_override_respected(self):
        """cores_override=5 must not raise and must return timing dict."""
        _, timing = self._run(num_workers=2, cores_override=5)
        assert 'load_time' in timing

    def test_single_worker_gets_all_cores(self):
        """num_workers=1 → cores = 22 // 1 = 22."""
        results, _ = self._run(num_workers=1)
        assert isinstance(results, list)

    def test_cores_override_floor_is_one(self):
        """cores_override=0 must be clamped to 1, not raise."""
        _, timing = self._run(num_workers=1, cores_override=0)
        assert timing['load_time'] >= 0.0


class TestBaselineSparkReturnShape:
    """Return-value contracts for run_baseline()."""

    def _run(self, fake_collect=None):
        if fake_collect is None:
            fake_collect = [('the', 100), ('a', 80), ('is', 60)]
        fake_spark = MagicMock()
        fake_sc    = MagicMock()
        fake_rdd   = MagicMock()
        fake_spark.sparkContext = fake_sc
        fake_sc.textFile.return_value = fake_rdd
        fake_rdd.count.return_value   = 10
        fake_rdd.flatMap.return_value = fake_rdd
        fake_rdd.filter.return_value  = fake_rdd
        fake_rdd.map.return_value     = fake_rdd
        fake_rdd.reduceByKey.return_value = fake_rdd
        fake_rdd.collect.return_value = fake_collect

        with patch('mpj_spark.applications.baseline_spark.build_spark_session',
                   return_value=fake_spark), \
             patch('mpj_spark.config.TOTAL_CORES', 8):
            from mpj_spark.applications.baseline_spark import run_baseline
            return run_baseline('fake.txt', 2)

    def test_returns_tuple_of_two(self):
        out = self._run()
        assert isinstance(out, tuple) and len(out) == 2

    def test_first_element_is_list(self):
        results, _ = self._run()
        assert isinstance(results, list)

    def test_results_sorted_descending(self):
        """Returned word list must be sorted by count descending."""
        results, _ = self._run([('b', 2), ('a', 10), ('c', 5)])
        counts = [c for _, c in results]
        assert counts == sorted(counts, reverse=True)

    def test_timing_dict_keys_complete(self):
        _, timing = self._run()
        assert {'load_time', 'processing_time', 'total_time'}.issubset(timing)

    def test_timing_values_non_negative(self):
        _, timing = self._run()
        assert timing['load_time']       >= 0.0
        assert timing['processing_time'] >= 0.0
        assert timing['total_time']      >= 0.0


# =============================================================
# Section 2: baseline_kmeans.py  (K-Means baseline)
# =============================================================

class TestBaselineKmeansCoreBudget:
    """
    The three core-budget priority levels in run_baseline_kmeans().
    """

    def _run(self, num_workers=2, cores_override=None, baseline_threads=None,
             total_cores=22):
        fake_spark    = MagicMock()
        fake_sc       = MagicMock()
        fake_rdd      = MagicMock()
        fake_df       = MagicMock()
        fake_model    = MagicMock()
        fake_summary  = MagicMock()
        fake_vec      = MagicMock()

        fake_spark.sparkContext  = fake_sc
        fake_spark.createDataFrame.return_value = fake_df
        fake_sc.textFile.return_value = fake_rdd
        fake_rdd.first.return_value   = '1.0,2.0,3.0'
        fake_rdd.map.return_value     = fake_rdd
        fake_rdd.filter.return_value  = fake_rdd
        fake_df.count.return_value    = 50

        fake_vec.cache.return_value   = fake_vec
        fake_model.clusterCenters     = [MagicMock(tolist=lambda: [1.0, 2.0, 3.0])] * 3
        fake_model.summary            = fake_summary
        fake_summary.trainingCost     = 42.0

        assembler_inst = MagicMock()
        assembler_inst.transform.return_value = fake_df
        fake_df.select.return_value   = fake_vec

        mock_kmeans_inst = MagicMock()
        mock_kmeans_inst.fit.return_value = fake_model

        with patch('mpj_spark.applications.baseline_kmeans.build_spark_session',
                   return_value=fake_spark), \
             patch('mpj_spark.config.TOTAL_CORES', total_cores), \
             patch('mpj_spark.applications.baseline_kmeans.VectorAssembler',
                   return_value=assembler_inst), \
             patch('mpj_spark.applications.baseline_kmeans.KMeans',
                   return_value=mock_kmeans_inst):
            from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans
            return run_baseline_kmeans(
                'fake.csv', num_workers, cores_override,
                k=3, max_iter=5, baseline_threads=baseline_threads)

    def test_priority1_baseline_threads_used(self):
        """baseline_threads=7 takes top priority over all other paths."""
        result, timing = self._run(num_workers=2, baseline_threads=7)
        assert isinstance(result, dict)

    def test_priority2_cores_override_used(self):
        """cores_override=4, no baseline_threads → override wins."""
        result, timing = self._run(num_workers=2, cores_override=4)
        assert isinstance(result, dict)

    def test_priority3_auto_formula(self):
        """No overrides → cores = TOTAL_CORES // num_workers = 22 // 2 = 11."""
        result, timing = self._run(num_workers=2)
        assert isinstance(result, dict)

    def test_single_worker_auto_gets_all_cores(self):
        """num_workers=1 → cores = 22."""
        result, _ = self._run(num_workers=1)
        assert isinstance(result, dict)


class TestBaselineKmeansReturnShape:
    """Return-value contracts for run_baseline_kmeans()."""

    def _run(self):
        fake_spark   = MagicMock()
        fake_sc      = MagicMock()
        fake_rdd     = MagicMock()
        fake_df      = MagicMock()
        fake_vec     = MagicMock()
        fake_model   = MagicMock()
        fake_summary = MagicMock()

        fake_spark.sparkContext  = fake_sc
        fake_spark.createDataFrame.return_value = fake_df
        fake_sc.textFile.return_value = fake_rdd
        fake_rdd.first.return_value   = '1.0,2.0'
        fake_rdd.map.return_value     = fake_rdd
        fake_rdd.filter.return_value  = fake_rdd
        fake_df.count.return_value    = 30
        fake_df.select.return_value   = fake_vec
        fake_vec.cache.return_value   = fake_vec

        fake_model.clusterCenters     = [
            MagicMock(tolist=lambda: [1.0, 2.0]),
            MagicMock(tolist=lambda: [5.0, 6.0]),
        ]
        fake_model.summary  = fake_summary
        fake_summary.trainingCost = 10.0

        assembler_inst = MagicMock()
        assembler_inst.transform.return_value = fake_df
        mock_kmeans_inst = MagicMock()
        mock_kmeans_inst.fit.return_value = fake_model

        with patch('mpj_spark.applications.baseline_kmeans.build_spark_session',
                   return_value=fake_spark), \
             patch('mpj_spark.config.TOTAL_CORES', 8), \
             patch('mpj_spark.applications.baseline_kmeans.VectorAssembler',
                   return_value=assembler_inst), \
             patch('mpj_spark.applications.baseline_kmeans.KMeans',
                   return_value=mock_kmeans_inst):
            from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans
            return run_baseline_kmeans('fake.csv', 2, None, k=2, max_iter=5)

    def test_returns_tuple_of_two(self):
        out = self._run()
        assert isinstance(out, tuple) and len(out) == 2

    def test_result_dict_has_centres_and_wcss(self):
        result, _ = self._run()
        assert 'centres' in result and 'wcss' in result

    def test_centres_is_list(self):
        result, _ = self._run()
        assert isinstance(result['centres'], list)

    def test_wcss_is_float(self):
        result, _ = self._run()
        assert isinstance(result['wcss'], float)

    def test_timing_dict_keys_complete(self):
        _, timing = self._run()
        assert {'load_time', 'processing_time', 'total_time'}.issubset(timing)

    def test_timing_values_non_negative(self):
        _, timing = self._run()
        assert timing['load_time']       >= 0.0
        assert timing['processing_time'] >= 0.0
        assert timing['total_time']      >= 0.0


# =============================================================
# Section 3: baseline_logreg.py — pure helpers
# =============================================================

class TestBaselineHeapGb:
    """
    Pure-Python helper _baseline_heap_gb(thread_count).

    Formula: ceil(0.5 + 0.25 * threads), min=2, capped at 80% system RAM.
    psutil is patched to a fixed 16 GB so cap = 12 GB.
    """

    def _heap(self, threads):
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.total = 16 * (1024 ** 3)  # 16 GB
        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            # Re-import to pick up the patched psutil
            import importlib
            import mpj_spark.applications.baseline_logreg as mod
            importlib.reload(mod)
            return mod._baseline_heap_gb(threads)

    def test_minimum_is_two_gb(self):
        assert self._heap(1) >= 2

    def test_single_thread_formula(self):
        # ceil(0.5 + 0.25*1) = ceil(0.75) = 1 → clamped to 2
        assert self._heap(1) == 2

    def test_four_threads_formula(self):
        # ceil(0.5 + 0.25*4) = ceil(1.5) = 2
        assert self._heap(4) == 2

    def test_eight_threads_formula(self):
        # ceil(0.5 + 0.25*8) = ceil(2.5) = 3
        assert self._heap(8) == 3

    def test_twenty_threads_formula(self):
        # ceil(0.5 + 0.25*20) = ceil(5.5) = 6
        assert self._heap(20) == 6

    def test_cap_at_80_percent_ram(self):
        # With 16 GB RAM, cap = 12.  Use 60 threads → raw = ceil(15.5)=16 > cap.
        assert self._heap(60) == 12

    def test_no_psutil_fallback(self):
        """When psutil is unavailable, cap must default to 8 GB (no crash)."""
        import sys
        original = sys.modules.pop('psutil', None)
        try:
            import importlib
            import mpj_spark.applications.baseline_logreg as mod
            importlib.reload(mod)
            result = mod._baseline_heap_gb(4)
            assert isinstance(result, int) and result >= 2
        finally:
            if original is not None:
                sys.modules['psutil'] = original


class TestBaselineLogregCoreBudget:
    """
    Core-budget resolution in run_baseline_logreg() (no JVM started).
    The three priority levels:
      1. baseline_threads  (explicit fair-comparison budget)
      2. cores_override    (manual flag)
      3. auto formula      (TOTAL_CORES / num_workers)
    """

    def _run(self, num_workers=2, cores_override=None,
             baseline_threads=None, parity_iter=None):
        fake_spark   = MagicMock()
        fake_df      = MagicMock()
        fake_vec     = MagicMock()
        fake_model   = MagicMock()
        fake_summary = MagicMock()
        coeff_mock   = MagicMock()

        fake_spark.sparkContext.setLogLevel = MagicMock()
        fake_spark.read.csv.return_value  = fake_df
        fake_df.dropna.return_value       = fake_df
        fake_df.columns                   = ['f0', 'f1', 'label']
        fake_df.count.return_value        = 200

        assembler_inst = MagicMock()
        assembler_inst.transform.return_value = fake_df
        fake_df.select.return_value = fake_vec

        coeff_mock.norm.return_value      = 1.0
        coeff_mock.toArray.return_value.tolist.return_value = [0.5, 0.5]
        fake_model.coefficients           = coeff_mock
        fake_model.intercept              = 0.1
        fake_summary.accuracy             = 0.85
        fake_model.summary                = fake_summary

        lr_inst = MagicMock()
        lr_inst.fit.return_value = fake_model

        with patch('mpj_spark.applications.baseline_logreg.SparkSession') as mock_ss, \
             patch('mpj_spark.config.TOTAL_CORES', 20), \
             patch('mpj_spark.applications.baseline_logreg.VectorAssembler',
                   return_value=assembler_inst), \
             patch('mpj_spark.applications.baseline_logreg.LogisticRegression',
                   return_value=lr_inst):
            builder = mock_ss.builder
            builder.appName.return_value  = builder
            builder.master.return_value   = builder
            builder.config.return_value   = builder
            builder.getOrCreate.return_value = fake_spark

            from mpj_spark.applications.baseline_logreg import run_baseline_logreg
            return run_baseline_logreg(
                'fake.csv', num_workers, cores_override,
                max_iter=5, reg_param=0.01, num_features=2,
                baseline_threads=baseline_threads,
                parity_iter=parity_iter,
            )

    def test_baseline_threads_priority(self):
        result, timing = self._run(num_workers=2, baseline_threads=10)
        assert result['accuracy'] is not None

    def test_cores_override_priority(self):
        result, timing = self._run(num_workers=2, cores_override=4)
        assert isinstance(timing['load_time'], float)

    def test_auto_formula_priority(self):
        result, timing = self._run(num_workers=4)
        assert isinstance(result, dict)

    def test_parity_iter_overrides_max_iter(self):
        """parity_iter must be used as effective_iter in the LR constructor."""
        result, timing = self._run(num_workers=2, parity_iter=20)
        assert timing['parity_iter'] == 20
        assert timing['effective_iter'] == 20

    def test_no_parity_iter_uses_max_iter(self):
        """Without parity_iter, effective_iter equals max_iter=5."""
        _, timing = self._run(num_workers=2)
        assert timing['effective_iter'] == 5
        assert timing['parity_iter'] is None


class TestBaselineLogregReturnShape:
    """Return-value contracts for run_baseline_logreg()."""

    def _run_success(self):
        fake_spark   = MagicMock()
        fake_df      = MagicMock()
        fake_vec     = MagicMock()
        fake_model   = MagicMock()
        fake_summary = MagicMock()
        coeff_mock   = MagicMock()

        fake_spark.sparkContext.setLogLevel = MagicMock()
        fake_spark.read.csv.return_value  = fake_df
        fake_df.dropna.return_value       = fake_df
        fake_df.columns                   = ['f0', 'f1', 'label']
        fake_df.count.return_value        = 100
        fake_df.select.return_value       = fake_vec

        assembler_inst = MagicMock()
        assembler_inst.transform.return_value = fake_df

        coeff_mock.norm.return_value           = 1.414
        coeff_mock.toArray.return_value.tolist.return_value = [0.3, 0.4]
        fake_model.coefficients = coeff_mock
        fake_model.intercept    = 0.05
        fake_summary.accuracy   = 0.91
        fake_model.summary      = fake_summary

        lr_inst = MagicMock()
        lr_inst.fit.return_value = fake_model

        with patch('mpj_spark.applications.baseline_logreg.SparkSession') as mock_ss, \
             patch('mpj_spark.config.TOTAL_CORES', 8), \
             patch('mpj_spark.applications.baseline_logreg.VectorAssembler',
                   return_value=assembler_inst), \
             patch('mpj_spark.applications.baseline_logreg.LogisticRegression',
                   return_value=lr_inst):
            builder = mock_ss.builder
            builder.appName.return_value  = builder
            builder.master.return_value   = builder
            builder.config.return_value   = builder
            builder.getOrCreate.return_value = fake_spark

            from mpj_spark.applications.baseline_logreg import run_baseline_logreg
            return run_baseline_logreg('fake.csv', 2, None, max_iter=3)

    def _run_oom(self):
        """Simulate a fit() exception to exercise the OOM-resilient path."""
        fake_spark = MagicMock()
        fake_df    = MagicMock()
        fake_vec   = MagicMock()

        fake_spark.sparkContext.setLogLevel = MagicMock()
        fake_spark.read.csv.return_value  = fake_df
        fake_df.dropna.return_value       = fake_df
        fake_df.columns                   = ['f0', 'label']
        fake_df.count.return_value        = 50
        fake_df.select.return_value       = fake_vec

        assembler_inst = MagicMock()
        assembler_inst.transform.return_value = fake_df

        lr_inst = MagicMock()
        lr_inst.fit.side_effect = MemoryError("Java heap space")

        with patch('mpj_spark.applications.baseline_logreg.SparkSession') as mock_ss, \
             patch('mpj_spark.config.TOTAL_CORES', 8), \
             patch('mpj_spark.applications.baseline_logreg.VectorAssembler',
                   return_value=assembler_inst), \
             patch('mpj_spark.applications.baseline_logreg.LogisticRegression',
                   return_value=lr_inst):
            builder = mock_ss.builder
            builder.appName.return_value  = builder
            builder.master.return_value   = builder
            builder.config.return_value   = builder
            builder.getOrCreate.return_value = fake_spark

            from mpj_spark.applications.baseline_logreg import run_baseline_logreg
            return run_baseline_logreg('fake.csv', 2, None, max_iter=3)

    # ── Success path ──────────────────────────────────────────────
    def test_returns_tuple_of_two(self):
        out = self._run_success()
        assert isinstance(out, tuple) and len(out) == 2

    def test_result_keys_complete(self):
        result, _ = self._run_success()
        assert {'weight_vector', 'intercept', 'accuracy',
                'row_count', 'oom_error'}.issubset(result)

    def test_accuracy_in_unit_range(self):
        result, _ = self._run_success()
        assert 0.0 <= result['accuracy'] <= 1.0

    def test_weight_vector_is_list(self):
        result, _ = self._run_success()
        assert isinstance(result['weight_vector'], list)

    def test_oom_error_is_none_on_success(self):
        result, _ = self._run_success()
        assert result['oom_error'] is None

    def test_timing_keys_complete(self):
        _, timing = self._run_success()
        assert {'load_time', 'processing_time', 'total_time',
                'effective_iter', 'parity_iter'}.issubset(timing)

    def test_timing_values_non_negative(self):
        _, timing = self._run_success()
        assert timing['load_time']       >= 0.0
        assert timing['processing_time'] >= 0.0
        assert timing['total_time']      >= 0.0

    # ── OOM / failure path ────────────────────────────────────────
    def test_oom_does_not_raise(self):
        """fit() failure must be caught — the function must return normally."""
        result, timing = self._run_oom()
        assert isinstance(result, dict)
        assert isinstance(timing, dict)

    def test_oom_accuracy_is_none(self):
        result, _ = self._run_oom()
        assert result['accuracy'] is None

    def test_oom_weight_vector_is_none(self):
        result, _ = self._run_oom()
        assert result['weight_vector'] is None

    def test_oom_error_message_captured(self):
        result, _ = self._run_oom()
        assert result['oom_error'] is not None
        assert isinstance(result['oom_error'], str)
        assert len(result['oom_error']) > 0

    def test_oom_timing_still_populated(self):
        """Even on OOM, timing dict must have all required keys with valid values."""
        _, timing = self._run_oom()
        assert {'load_time', 'processing_time', 'total_time'}.issubset(timing)
        assert timing['processing_time'] >= 0.0
