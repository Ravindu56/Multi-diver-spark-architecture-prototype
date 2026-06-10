# ================================================================
# mpj_spark/workers/worker_mpi.py  -  MPI Worker Runner  (ranks 1..N)
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
#
# PURPOSE
# -------
# MPI-native worker runner.  Called by every MPI rank != 0.
# Handles the MPI boot sequence (recv config, init SparkSession,
# send TAG_READY, recv TAG_GO) then delegates all Spark application
# logic to run_worker_core() in worker_process.py.
#
# Transport adapters (MpiWorkerAllreduceAdapter, MpiKMeansGossipAdapter,
# MpiReassignAdapter) wrap comm.send / comm.recv behind a Queue-compatible
# .put() / .get() interface so run_worker_core() remains fully transport-
# agnostic.
#
# P3-02 ACCEPTANCE CRITERION  (worker side)
# ------------------------------------------
#   MPI_COMM_WORLD replaces multiprocessing.Process; workers are ranks 1..N.
#   run_worker_core() contains zero multiprocessing or mpi4py imports.
#
# MPI BOOT SEQUENCE
# -----------------
#   recv(TAG_CONFIG)      <- root sends partition path + cfg dict
#   build_spark_session() <- JVM + SparkContext init
#   send(TAG_READY)       -> JVM-ready sentinel to root
#   recv(TAG_GO)          <- go-signal from root
#   run_worker_core()     <- all Spark logic
#   [adapters handle TAG_ALLREDUCE_UP/DOWN and TAG_REASSIGN_* internally]
#
# MPI TAG ALLOCATION  (consistent with root_mpi.py)
# --------------------------------------------------
#   TAG_CONFIG          = 10   root -> worker
#   TAG_RESULT          = 20   worker -> root
#   TAG_TIMING          = 21   worker -> root
#   TAG_ALLREDUCE_UP    = 30   worker -> root  (logreg weights / kmeans gossip)
#   TAG_ALLREDUCE_DOWN  = 31   root -> worker  (averaged weights)
#   TAG_REASSIGN_BCAST  = 40   root -> worker
#   TAG_REASSIGN_STATS  = 41   worker -> root
#   TAG_READY           = 50   worker -> root
#   TAG_GO              = 60   root -> worker
# ================================================================

import time
import traceback

from mpj_spark.utils.dev_logger import DevLogger
from mpj_spark.workers.spark_session import build_spark_session
from mpj_spark.workers.worker_process import _tag, run_worker_core

# ── MPI tag constants (mirrors root_mpi.py) ─────────────────────────
TAG_CONFIG = 10
TAG_RESULT = 20
TAG_TIMING = 21
TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31
TAG_REASSIGN_BCAST = 40
TAG_REASSIGN_STATS = 41
TAG_READY = 50
TAG_GO = 60


# ================================================================
# MPI Transport Adapters
# ================================================================
# Each adapter wraps a pair of MPI send/recv calls behind the
# Queue-compatible .put() / .get() interface expected by
# run_worker_core() and logreg.run().


class MpiWorkerAllreduceAdapter:
    """
    Queue-like adapter for LogReg per-iteration FedAvg.

    UP  (direction='up')   : .put(msg) -> comm.send(TAG_ALLREDUCE_UP)
    DOWN (direction='down'): .get()    -> comm.recv(TAG_ALLREDUCE_DOWN)
    """

    def __init__(self, comm, direction: str):
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")
        self._comm = comm
        self._direction = direction

    def put(self, msg, block=True, timeout=None):
        if self._direction != "up":
            raise RuntimeError("put() called on DOWN adapter — use get().")
        self._comm.send(msg, dest=0, tag=TAG_ALLREDUCE_UP)

    def get(self, block=True, timeout=None):
        if self._direction != "down":
            raise RuntimeError("get() called on UP adapter — use put().")
        return self._comm.recv(source=0, tag=TAG_ALLREDUCE_DOWN)

    def empty(self):
        return False

    def qsize(self):
        return 0


class MpiKMeansGossipAdapter:
    """
    Queue-like adapter for K-Means gossip (one .put() after local fit).
    """

    def __init__(self, comm):
        self._comm = comm

    def put(self, msg, block=True, timeout=None):
        self._comm.send(msg, dest=0, tag=TAG_ALLREDUCE_UP)

    def empty(self):
        return True

    def qsize(self):
        return 0


class MpiReassignAdapter:
    """
    Queue-like adapter for the K-Means re-assignment pass.

    .get()    <- comm.recv(TAG_REASSIGN_BCAST)  receives global centroids
    .put(msg) -> comm.send(TAG_REASSIGN_STATS)  sends cluster sums/counts
    """

    def __init__(self, comm):
        self._comm = comm

    def get(self, block=True, timeout=None):
        return self._comm.recv(source=0, tag=TAG_REASSIGN_BCAST)

    def put(self, msg, block=True, timeout=None):
        self._comm.send(msg, dest=0, tag=TAG_REASSIGN_STATS)

    def empty(self):
        return False

    def qsize(self):
        return 0


# ================================================================
# run_worker_mpi  —  main MPI worker entry point
# ================================================================


def run_worker_mpi(comm):
    """
    MPI-native worker runner.  Must be called by every rank != 0.

    Handles the MPI boot sequence then delegates all Spark logic to
    run_worker_core() via MPI transport adapters.

    P3-02: MPI_COMM_WORLD replaces multiprocessing.Process.
    run_worker_core() contains zero multiprocessing or mpi4py imports.
    """
    rank = comm.Get_rank()
    worker_id = rank - 1  # 0-indexed: rank 1 ⇒ worker 0, rank 2 ⇒ worker 1

    assert rank != 0, "run_worker_mpi() must not be called by rank 0."

    # ================================================================
    # Step 1 — Receive configuration from root (TAG_CONFIG)
    # ================================================================
    print(f"{_tag(worker_id, 'BOOT')} rank={rank}  waiting for config ...")
    cfg = comm.recv(source=0, tag=TAG_CONFIG)

    partition_path = cfg["partition_path"]
    app_name = cfg.get("app", "wordcount")
    cores_override = cfg.get("cores_override", None)
    num_workers = cfg.get("num_workers", 1)

    print(
        f"{_tag(worker_id, 'BOOT')} config received  " f"app={app_name}  partition={partition_path}"
    )

    logger = DevLogger(worker_id=worker_id)

    # ================================================================
    # Step 2 — Initialise SparkSession
    # ================================================================
    print(f"{_tag(worker_id, 'INIT')} Starting SparkSession (app={app_name}) ...")
    t_init_start = time.perf_counter()

    try:
        spark = build_spark_session(
            app_name=f"MPJ-MPI-Worker-{worker_id}-{app_name}",
            cores_override=cores_override,
            num_workers=num_workers,
        )
    except Exception as exc:
        print(f"{_tag(worker_id, 'INIT')} SparkSession FAILED: {exc}")
        traceback.print_exc()
        # Unblock root’s recv loop even on init failure
        comm.send("ready", dest=0, tag=TAG_READY)
        comm.recv(source=0, tag=TAG_GO)
        comm.send(
            {
                "worker_id": worker_id,
                "result": None,
                "status": "error",
                "error": str(exc),
            },
            dest=0,
            tag=TAG_RESULT,
        )
        comm.send(
            {
                "worker_id": worker_id,
                "init_time": 0.0,
                "load_time": 0.0,
                "processing_time": 0.0,
                "total_time": 0.0,
            },
            dest=0,
            tag=TAG_TIMING,
        )
        return

    init_time = time.perf_counter() - t_init_start
    print(f"{_tag(worker_id, 'INIT')} SparkSession ready  ({init_time:.3f}s)")

    # ================================================================
    # Step 3 — Signal JVM-ready to root (replaces ready_signal.set())
    # ================================================================
    comm.send("ready", dest=0, tag=TAG_READY)
    print(f"{_tag(worker_id, 'WAIT')} JVM-ready sent — waiting for go-signal ...")

    # ================================================================
    # Step 4 — Wait for simultaneous go-signal (replaces go_signal.wait())
    # ================================================================
    comm.recv(source=0, tag=TAG_GO)
    print(f"{_tag(worker_id, 'WAIT')} Go-signal received — starting {app_name}")

    # ================================================================
    # Step 5 — Build MPI transport adapters, delegate to core
    # ================================================================
    up_queue = None
    down_queue = None
    reassign = None

    if app_name == "kmeans":
        up_queue = MpiKMeansGossipAdapter(comm)
        reassign = MpiReassignAdapter(comm)
    elif app_name == "logreg":
        up_queue = MpiWorkerAllreduceAdapter(comm, direction="up")
        down_queue = MpiWorkerAllreduceAdapter(comm, direction="down")

    outcome = run_worker_core(
        worker_id=worker_id,
        partition_path=partition_path,
        spark=spark,
        worker_config=cfg,
        up_queue=up_queue,
        down_queue=down_queue,
        reassign_adapter=reassign,
    )

    # Patch init_time into the timing dict (core doesn’t know it)
    outcome["timing"]["init_time"] = init_time
    outcome["timing"]["total_time"] = (
        init_time + outcome["timing"]["load_time"] + outcome["timing"]["processing_time"]
    )

    # Re-log with correct init_time
    logger.log_worker_timing(
        worker_id=worker_id,
        init_time=init_time,
        load_time=outcome["timing"]["load_time"],
        proc_time=outcome["timing"]["processing_time"],
    )

    # ================================================================
    # Step 6 — Send result and timing to root via MPI
    # ================================================================
    comm.send(
        {
            "worker_id": worker_id,
            "result": outcome["result"],
            "status": outcome["status"],
            **({"error": outcome["error"]} if outcome["status"] == "error" else {}),
        },
        dest=0,
        tag=TAG_RESULT,
    )
    comm.send(outcome["timing"], dest=0, tag=TAG_TIMING)

    # ================================================================
    # Teardown
    # ================================================================
    try:
        spark.stop()
        print(f"{_tag(worker_id, 'STOP')} SparkSession stopped.")
    except Exception:
        pass
