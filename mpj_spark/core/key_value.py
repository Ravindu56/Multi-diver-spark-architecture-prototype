# ============================================================
# core/key_value.py
# KeyValueStructure — RDD ↔ MPJ message-passing buffer
# Paper Reference: Section IV.D — Data Conversion
# ============================================================


class KeyValueStructure:
    """
    Converts Spark RDD collect() results into contiguous (key, value)
    arrays suitable for serialisation over Python multiprocessing Queue
    (simulates MPJ message-passing buffer).
    """

    def __init__(self):
        self.data: list = []   # list of (str, int) tuples

    # ----------------------------------------------------------
    def from_rdd_collect(self, rdd_results):
        """Populate from an RDD.collect() call."""
        self.data = list(rdd_results)
        return self

    def to_serializable(self) -> list:
        """Convert to JSON-safe list for Queue transmission."""
        return [(str(k), int(v)) for k, v in self.data]

    @staticmethod
    def from_serializable(serialized: list) -> 'KeyValueStructure':
        """Reconstruct from a serialised list received via Queue."""
        kv = KeyValueStructure()
        kv.data = [(str(k), int(v)) for k, v in serialized]
        return kv
