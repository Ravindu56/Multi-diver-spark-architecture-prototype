# =============================================================
# tests/unit/test_key_value.py
#
# Unit tests for mpj_spark/core/key_value.py (KeyValueStructure)
# Covers Phase 1: RDD ↔ Queue serialisation and WordCount merge.
#
# Research alignment:
#   - Objective 1c: correctness of WordCount batch analytics workload
#   - Objective 1a: data conversion between Spark RDD and MPI buffer
# =============================================================
from mpj_spark.core.key_value import KeyValueStructure


# =============================================================
# Section 1: Construction helpers
# =============================================================


class TestFromRddCollect:
    """Tests for from_rdd_collect() — RDD result ingestion."""

    def test_populates_data(self):
        kv = KeyValueStructure().from_rdd_collect([("hello", 3), ("world", 5)])
        assert len(kv.data) == 2

    def test_preserves_order(self):
        pairs = [("apple", 1), ("banana", 2), ("cherry", 3)]
        kv = KeyValueStructure().from_rdd_collect(pairs)
        assert kv.data == pairs

    def test_empty_rdd(self):
        kv = KeyValueStructure().from_rdd_collect([])
        assert kv.data == []

    def test_returns_self(self):
        """from_rdd_collect must return the instance for method chaining."""
        kv = KeyValueStructure()
        result = kv.from_rdd_collect([("a", 1)])
        assert result is kv


# =============================================================
# Section 2: Serialisation round-trip
# =============================================================


class TestSerialisationRoundTrip:
    """
    Tests for to_serializable() and from_serializable().
    These simulate Queue transmission between root and workers.
    """

    def test_to_serializable_types(self):
        """to_serializable must return list of (str, int) pairs."""
        kv = KeyValueStructure().from_rdd_collect([("foo", 7)])
        result = kv.to_serializable()
        assert isinstance(result[0][0], str)
        assert isinstance(result[0][1], int)

    def test_round_trip_preserves_data(self):
        """
        Serialise → transmit over Queue → deserialise must recover
        the original (key, count) pairs exactly.
        """
        original = [("spark", 10), ("hadoop", 4), ("mpi", 7)]
        kv = KeyValueStructure().from_rdd_collect(original)
        serialized = kv.to_serializable()
        recovered = KeyValueStructure.from_serializable(serialized)
        assert sorted(recovered.data) == sorted(original)

    def test_from_serializable_empty(self):
        kv = KeyValueStructure.from_serializable([])
        assert kv.data == []

    def test_to_serializable_empty(self):
        kv = KeyValueStructure()
        assert kv.to_serializable() == []


# =============================================================
# Section 3: merge() — WordCount aggregation (Phase 1 core)
# =============================================================


class TestMerge:
    """
    Tests for merge() — the root-side WordCount aggregation step.
    This simulates collecting partial results from N drivers and
    combining them into a single global word count.
    """

    def test_merge_two_kv_objects_sums_counts(self):
        """Shared keys must be summed; unique keys must be preserved."""
        kv1 = KeyValueStructure().from_rdd_collect([("hello", 3), ("world", 2)])
        kv2 = KeyValueStructure().from_rdd_collect([("hello", 1), ("spark", 5)])
        kv1.merge(kv2)
        result = dict(kv1.data)
        assert result["hello"] == 4
        assert result["world"] == 2
        assert result["spark"] == 5

    def test_merge_with_raw_serializable_list(self):
        """
        merge() must also accept a raw serializable list (the output of
        to_serializable()) without requiring explicit deserialisation.
        This matches how root_process.py calls it.
        """
        kv = KeyValueStructure().from_rdd_collect([("foo", 10)])
        raw_list = [("foo", 5), ("bar", 3)]
        kv.merge(raw_list)
        result = dict(kv.data)
        assert result["foo"] == 15
        assert result["bar"] == 3

    def test_merge_empty_partition_leaves_data_unchanged(self):
        """
        Merging an empty partition (a worker that processed no words)
        must not alter the accumulated result.
        """
        kv = KeyValueStructure().from_rdd_collect([("hello", 5)])
        kv.merge(KeyValueStructure())
        assert dict(kv.data) == {"hello": 5}

    def test_merge_into_empty_base(self):
        """Merging a non-empty KV into an empty base must populate it."""
        base = KeyValueStructure()
        other = KeyValueStructure().from_rdd_collect([("mpi", 8)])
        base.merge(other)
        assert dict(base.data) == {"mpi": 8}

    def test_merge_three_workers_simulated(self):
        """
        Simulate 3 Spark drivers returning partial word counts.
        After sequential merge into root KV, counts must match
        the expected global totals.
        """
        workers = [
            [("the", 10), ("is", 5)],
            [("the", 8), ("a", 3)],
            [("is", 2), ("a", 4)],
        ]
        root_kv = KeyValueStructure()
        for w in workers:
            root_kv.merge(KeyValueStructure().from_rdd_collect(w))
        result = dict(root_kv.data)
        assert result["the"] == 18  # 10 + 8
        assert result["is"] == 7  # 5 + 2
        assert result["a"] == 7  # 3 + 4

    def test_merge_returns_self(self):
        """merge() must return self for chaining support."""
        kv = KeyValueStructure().from_rdd_collect([("x", 1)])
        result = kv.merge(KeyValueStructure())
        assert result is kv

    def test_merge_no_duplicate_keys_in_result(self):
        """After merge, each key must appear exactly once in data."""
        kv1 = KeyValueStructure().from_rdd_collect([("k", 3)])
        kv2 = KeyValueStructure().from_rdd_collect([("k", 7)])
        kv1.merge(kv2)
        keys = [k for k, _ in kv1.data]
        assert len(keys) == len(set(keys)), "Duplicate keys found after merge"


# =============================================================
# Section 4: get_top_n()
# =============================================================


class TestGetTopN:
    """Tests for get_top_n() — top-N word result extraction."""

    def test_returns_correct_n(self):
        kv = KeyValueStructure().from_rdd_collect(
            [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
        )
        assert len(kv.get_top_n(3)) == 3

    def test_sorted_descending(self):
        """Results must be ordered highest count first."""
        kv = KeyValueStructure().from_rdd_collect(
            [("low", 1), ("mid", 5), ("high", 10)]
        )
        result = kv.get_top_n(3)
        counts = [v for _, v in result]
        assert counts == sorted(counts, reverse=True)

    def test_top_1_returns_most_frequent(self):
        kv = KeyValueStructure().from_rdd_collect(
            [("rare", 1), ("common", 100), ("medium", 10)]
        )
        assert kv.get_top_n(1)[0] == ("common", 100)

    def test_n_larger_than_data_returns_all(self):
        """Requesting more than available items must return all items."""
        kv = KeyValueStructure().from_rdd_collect([("x", 1), ("y", 2)])
        assert len(kv.get_top_n(100)) == 2

    def test_top_n_empty_data(self):
        kv = KeyValueStructure()
        assert kv.get_top_n(5) == []
