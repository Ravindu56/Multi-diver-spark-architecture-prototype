# ============================================================
# core/key_value.py
# KeyValueStructure — RDD ↔ MPJ message-passing buffer
# Paper Reference: Section IV.D — Data Conversion
# ============================================================

from collections import defaultdict


class KeyValueStructure:
    """
    Converts Spark RDD collect() results into contiguous (key, value)
    arrays suitable for serialisation over Python multiprocessing Queue
    (simulates MPJ message-passing buffer).
    """

    def __init__(self):
        self.data: list = []  # list of (str, int) tuples

    # ----------------------------------------------------------
    def from_rdd_collect(self, rdd_results):
        """Populate from an RDD.collect() call."""
        self.data = list(rdd_results)
        return self

    def to_serializable(self) -> list:
        """Convert to JSON-safe list for Queue transmission."""
        return [(str(k), int(v)) for k, v in self.data]

    @staticmethod
    def from_serializable(serialized: list) -> "KeyValueStructure":
        """Reconstruct from a serialised list received via Queue."""
        kv = KeyValueStructure()
        kv.data = [(str(k), int(v)) for k, v in serialized]
        return kv

    def merge(self, other) -> "KeyValueStructure":
        """
        Merge another KeyValueStructure or a raw serializable list into
        this instance by summing counts for matching keys.

        Accepts both KeyValueStructure objects and plain lists (the output
        of to_serializable()) so that root_process.py can call merge()
        directly on Queue-received worker results without explicit
        deserialization.
        """
        combined = defaultdict(int)
        for k, v in self.data:
            combined[k] += v
        other_data = other.data if isinstance(other, KeyValueStructure) else other
        for k, v in other_data:
            combined[k] += v
        self.data = list(combined.items())
        return self

    def get_top_n(self, n: int) -> list:
        """Return the top-N (key, value) pairs sorted by count descending."""
        return sorted(self.data, key=lambda x: x[1], reverse=True)[:n]
