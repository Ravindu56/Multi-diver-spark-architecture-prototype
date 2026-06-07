# ================================================================
# mpj_spark_mpi.py  -  Phase 3 MPI Transport Wrapper
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
#
# PURPOSE OF THIS FILE
# --------------------
# This file is a *transport wrapper only*. It contains zero application
# logic. All business logic lives in the mpj_spark package:
#
#   mpj_spark/core/root_process.py      -> run_root()
#   mpj_spark/workers/worker_process.py -> worker_process()
#   mpj_spark/workers/spark_session.py  -> build_spark_session()
#   mpj_spark/core/file_manager.py      -> MPJSparkFileManager
#   mpj_spark/core/key_value.py         -> KeyValueStructure
#
# WHAT THIS FILE ADDS
# -------------------
#   MpiQueue  - drop-in multiprocessing.Queue replacement over MPI
#   MpiEvent  - drop-in multiprocessing.Event replacement over MPI
#   root_main()   - injects MPI transport into run_root()
#   worker_main() - injects MPI transport into worker_process()
#
# LAUNCH
# ------
#   mpirun --oversubscribe -np <1+N> python3 mpj_spark_mpi.py [options]
#   Rank 0  = root coordinator
#   Rank 1+ = independent Spark driver workers
# ================================================================

import os
import sys

os.environ.setdefault(
    "JAVA_TOOL_OPTIONS",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "-Djava.security.manager=allow",
)
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import argparse  # noqa: E402
from mpi4py import MPI  # noqa: E402

# ── MPI communicator globals ──────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()   # 1 root + N workers


# ================================================================
# MPI TRANSPORT ADAPTERS
# ================================================================
# These classes implement the same interface as multiprocessing.Queue
# and multiprocessing.Event so that mpj_spark package code (which was
# written for Phase 2 multiprocessing) runs unchanged under MPI.
#
# Tag allocation (avoids message collisions):
#   10   partition metadata  root -> worker
#   20   result payload      worker -> root
#   21   timing payload      worker -> root
#   30   gossip / allreduce-UP   worker -> root
#   31   allreduce-DOWN      root -> worker
#   40   reassign broadcast  root -> worker
#   41   reassign stats      worker -> root
#   50   ready signal        worker -> root
#   60   go signal           root -> worker
# ================================================================

class MpiQueue:
    """
    multiprocessing.Queue-compatible adapter.

    Each MpiQueue is bound to a fixed MPI tag so that separate
    logical channels (result, timing, gossip, allreduce) never
    interfere with each other.

    put(obj)          -> comm.send(obj, dest=peer, tag=self.tag)
    get(timeout=None) -> comm.recv(source=peer, tag=self.tag)

    For the root, peer=worker_rank (set per-worker at runtime).
    For workers, peer=0 (always the root).
    """

    def __init__(self, tag: int, peer: int):
        self._tag  = tag
        self._peer = peer

    # ── Queue interface ───────────────────────────────────────────
    def put(self, obj, block=True, timeout=None):
        comm.send(obj, dest=self._peer, tag=self._tag)

    def get(self, block=True, timeout=None):
        # mpi4py recv honours source/tag matching; timeout not natively
        # supported but acceptable for prototype (all workers expected
        # to complete within the Spark job timeout)
        return comm.recv(source=self._peer, tag=self._tag)

    def empty(self):
        return not comm.Iprobe(source=self._peer, tag=self._tag)

    def qsize(self):
        raise NotImplementedError


class MpiRootFanoutQueue:
    """
    Root-side broadcast queue: put() sends to ALL workers; get()
    collects one message from each worker in rank order.

    Used for channels where root must hear from every worker
    (result_queue, timing_queue) or broadcast to every worker
    (allreduce_down_queue, go_signal broadcast).
    """

    def __init__(self, tag: int, num_workers: int):
        self._tag         = tag
        self._num_workers = num_workers
        self._recv_buf    = []     # pre-collected messages

    def put(self, obj, block=True, timeout=None):
        """Broadcast obj to all workers."""
        for w in range(1, self._num_workers + 1):
            comm.send(obj, dest=w, tag=self._tag)

    def get(self, block=True, timeout=None):
        """Return next buffered message, collecting from all workers first."""
        if not self._recv_buf:
            for w in range(1, self._num_workers + 1):
                self._recv_buf.append(comm.recv(source=w, tag=self._tag))
        return self._recv_buf.pop(0)

    def empty(self):
        return len(self._recv_buf) == 0 and not any(
            comm.Iprobe(source=w, tag=self._tag)
            for w in range(1, self._num_workers + 1)
        )


class MpiEvent:
    """
    multiprocessing.Event-compatible adapter.

    set()  -> send a sentinel to peer
    wait() -> blocking recv from peer

    Used for:
      ready_signal[i]  (worker -> root, tag=50): worker signals JVM ready
      go_signal[i]     (root -> worker, tag=60): root fires worker
    """

    def __init__(self, tag: int, peer: int):
        self._tag  = tag
        self._peer = peer
        self._flag = False

    def set(self):
        comm.send(True, dest=self._peer, tag=self._tag)
        self._flag = True

    def wait(self, timeout=None):
        if not self._flag:
            comm.recv(source=self._peer, tag=self._tag)
            self._flag = True

    def is_set(self):
        return self._flag

    def clear(self):
        self._flag = False


# ================================================================
# ROOT COORDINATOR  (rank 0)
# ================================================================

def root_main(args):
    """
    Injects MPI transport objects into run_root() from the package.

    run_root() was written for Phase 2 multiprocessing.Queue / Event.
    Here we replace every Queue and Event with MPI-backed adapters
    so the exact same coordination logic executes over MPI.
    """
    from mpj_spark.core.root_process import run_root

    num_workers = size - 1

    # ── Partition metadata: send directly, not via Queue ─────────────
    # run_root() creates the partitions and sends metadata via a
    # dedicated send loop inside itself. We pass partition_queues so
    # it can send metadata to each worker rank.
    partition_queues = [
        MpiQueue(tag=10, peer=w_rank)
        for w_rank in range(1, size)
    ]

    # ── Result and timing collection ──────────────────────────────────
    result_queue = MpiRootFanoutQueue(tag=20, num_workers=num_workers)
    timing_queue = MpiRootFanoutQueue(tag=21, num_workers=num_workers)

    # ── Barrier: ready_signals[i] receives "ready" from worker rank i+1
    ready_signals = [
        MpiEvent(tag=50, peer=w_rank)
        for w_rank in range(1, size)
    ]

    # ── Fire: go_signals[i] sends "go" to worker rank i+1
    go_signals = [
        MpiEvent(tag=60, peer=w_rank)
        for w_rank in range(1, size)
    ]

    # ── Optional channels (app-dependent) ────────────────────────────
    gossip_queue         = MpiRootFanoutQueue(tag=30, num_workers=num_workers)
    allreduce_down_queue = MpiRootFanoutQueue(tag=31, num_workers=num_workers)
    reassign_queue       = MpiRootFanoutQueue(tag=40, num_workers=num_workers)

    run_root(
        input_file           = args.input,
        num_workers          = num_workers,
        compare              = args.compare,
        prewarm              = not args.no_prewarm,
        cores_override       = args.cores,
        app                  = args.app,
        kmeans_k             = args.kmeans_k,
        kmeans_iter          = args.kmeans_iter,
        baseline_threads     = args.baseline_threads,
        use_gossip           = args.gossip,
        gossip_threshold     = args.gossip_threshold,
        gossip_max_rounds    = args.gossip_max_rounds,
        gossip_fanout        = args.gossip_fanout,
        use_global_seed      = args.global_seed,
        use_reassign         = args.reassign,
        logreg_iter          = args.logreg_iter,
        logreg_reg_param     = args.logreg_reg_param,
        logreg_features      = args.logreg_features,
        results_dir          = args.results_dir,
        # MPI transport injection:
        _partition_queues    = partition_queues,
        _result_queue        = result_queue,
        _timing_queue        = timing_queue,
        _ready_signals       = ready_signals,
        _go_signals          = go_signals,
        _gossip_queue        = gossip_queue,
        _allreduce_down_queue= allreduce_down_queue,
        _reassign_queue      = reassign_queue,
    )


# ================================================================
# WORKER  (rank >= 1)
# ================================================================

def worker_main():
    """
    Injects MPI transport objects into worker_process() from the package.

    worker_process() was written for Phase 2 multiprocessing.Queue / Event.
    Here we replace every Queue and Event with MPI-backed adapters.
    """
    from mpj_spark.workers.worker_process import worker_process

    # ── Receive partition path from root ──────────────────────────────
    # Root sends the partition path dict via tag=10
    partition_metadata = comm.recv(source=0, tag=10)
    partition_path     = partition_metadata["partition_path"]

    # ── Receive worker config from root ───────────────────────────────
    # Root broadcasts worker_cfg to all workers after partitioning
    worker_config = comm.bcast(None, root=0)

    # ── MPI-backed Queue / Event adapters ────────────────────────────
    result_queue         = MpiQueue(tag=20, peer=0)
    timing_queue         = MpiQueue(tag=21, peer=0)
    gossip_queue         = MpiQueue(tag=30, peer=0)   # allreduce-UP for logreg
    allreduce_down_queue = MpiQueue(tag=31, peer=0)   # allreduce-DOWN from root
    reassign_queue       = MpiQueue(tag=40, peer=0)

    # ready_signal: worker signals root it is JVM-ready (tag=50)
    ready_signal = MpiEvent(tag=50, peer=0)

    # go_signal: worker waits for root to fire (tag=60)
    go_signal = MpiEvent(tag=60, peer=0)

    # worker_id is rank - 1 (0-indexed, consistent with Phase 2)
    worker_id = rank - 1

    worker_process(
        worker_id            = worker_id,
        partition_path       = partition_path,
        result_queue         = result_queue,
        go_signal            = go_signal,
        ready_signal         = ready_signal,
        timing_queue         = timing_queue,
        worker_config        = worker_config,
        gossip_queue         = gossip_queue,
        reassign_queue       = reassign_queue,
        allreduce_down_queue = allreduce_down_queue,
    )


# ================================================================
# ENTRY POINT  -  rank-based dispatch
# ================================================================

if __name__ == "__main__":

    if rank == 0:
        # ── CLI (root only) ───────────────────────────────────────────
        parser = argparse.ArgumentParser(
            description="MPJ-SPARK Phase 3 - mpi4py multi-driver entry point"
        )
        parser.add_argument("--input",             default="./test_dataset.txt")
        parser.add_argument("--generate",           type=int, default=50,
                            help="Generate N MB synthetic dataset if --input not found")
        parser.add_argument("--workers",            type=int, default=size - 1)
        parser.add_argument("--app",                default="wordcount",
                            choices=["wordcount", "kmeans", "logreg"])
        parser.add_argument("--compare",            action="store_true")
        parser.add_argument("--no-prewarm",         action="store_true")
        parser.add_argument("--cores",              type=int, default=None)
        parser.add_argument("--kmeans-k",           type=int, default=3)
        parser.add_argument("--kmeans-iter",        type=int, default=20)
        parser.add_argument("--baseline-threads",   type=int, default=None)
        parser.add_argument("--gossip",             action="store_true")
        parser.add_argument("--gossip-threshold",   type=float, default=1e-3)
        parser.add_argument("--gossip-max-rounds",  type=int,   default=10)
        parser.add_argument("--gossip-fanout",      type=int,   default=2)
        parser.add_argument("--global-seed",        action="store_true")
        parser.add_argument("--reassign",           action="store_true")
        parser.add_argument("--logreg-iter",        type=int,   default=10)
        parser.add_argument("--logreg-reg-param",   type=float, default=0.01)
        parser.add_argument("--logreg-features",    type=int,   default=10)
        parser.add_argument("--results-dir",        default="results")
        args = parser.parse_args()

        # Auto-generate dataset if needed
        if not os.path.exists(args.input):
            from mpj_spark_prototype_v2 import generate_test_dataset
            args.input = generate_test_dataset(args.input, args.generate)

        root_main(args)

    else:
        # Workers have no CLI args to parse — go straight to work
        worker_main()
