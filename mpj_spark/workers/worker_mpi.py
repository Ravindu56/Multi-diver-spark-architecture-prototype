# ================================================================
# mpj_spark/workers/worker_mpi.py  -  MPI Worker Runner  (ranks 1..N)
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
# ================================================================

import time
import traceback

from mpj_spark.core.sync_modes import MODE_PS_SYNC_FEDAVG_MPI, normalize_sync_mode
from mpj_spark.utils.dev_logger import DevLogger
from mpj_spark.workers.spark_session import build_spark_session
from mpj_spark.workers.worker_process import _tag, run_worker_core

TAG_CONFIG = 10
TAG_RESULT = 20
TAG_TIMING = 21
TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31
TAG_REASSIGN_BCAST = 40
TAG_REASSIGN_STATS = 41
TAG_READY = 50
TAG_GO = 60


class MpiWorkerAllreduceAdapter:
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
    def __init__(self, comm):
        self._comm = comm

    def put(self, msg, block=True, timeout=None):
        self._comm.send(msg, dest=0, tag=TAG_ALLREDUCE_UP)

    def empty(self):
        return True

    def qsize(self):
        return 0


class MpiReassignAdapter:
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


def run_worker_mpi(comm):
    rank = comm.Get_rank()
    worker_id = rank - 1

    assert rank != 0, "run_worker_mpi() must not be called by rank 0."

    print(f"{_tag(worker_id, 'BOOT')} rank={rank} waiting for config ...")
    cfg = comm.recv(source=0, tag=TAG_CONFIG)

    partition_path = cfg["partition_path"]
    app_name = cfg.get("app", "wordcount")
    cores_override = cfg.get("cores_override", None)
    num_workers = cfg.get("num_workers", 1)
    sync_mode = normalize_sync_mode(cfg.get("sync_mode", MODE_PS_SYNC_FEDAVG_MPI))

    print(
        f"{_tag(worker_id, 'BOOT')} config received app={app_name} partition={partition_path} sync_mode={sync_mode}"
    )

    logger = DevLogger(worker_id=worker_id)

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
        comm.send("ready", dest=0, tag=TAG_READY)
        comm.recv(source=0, tag=TAG_GO)
        comm.send(
            {"worker_id": worker_id, "result": None, "status": "error", "error": str(exc)},
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
    print(f"{_tag(worker_id, 'INIT')} SparkSession ready ({init_time:.3f}s)")

    comm.send("ready", dest=0, tag=TAG_READY)
    print(f"{_tag(worker_id, 'WAIT')} JVM-ready sent — waiting for go-signal ...")

    comm.recv(source=0, tag=TAG_GO)
    print(f"{_tag(worker_id, 'WAIT')} Go-signal received — starting {app_name}")

    worker_comm = comm.Split(color=1, key=rank)

    up_queue = None
    down_queue = None
    reassign = None

    if app_name == "kmeans":
        up_queue = MpiKMeansGossipAdapter(comm)
        reassign = MpiReassignAdapter(comm)
    elif app_name == "logreg":
        if sync_mode == "ps_sync_fedavg_queue":
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
        comm=worker_comm,
        root_comm=comm,  # P3-09: COMM_WORLD channel for the async-PS P2P protocol
    )

    outcome["timing"]["init_time"] = init_time
    outcome["timing"]["total_time"] = (
        init_time + outcome["timing"]["load_time"] + outcome["timing"]["processing_time"]
    )

    logger.log_worker_timing(
        worker_id=worker_id,
        init_time=init_time,
        load_time=outcome["timing"]["load_time"],
        proc_time=outcome["timing"]["processing_time"],
    )

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

    try:
        spark.stop()
        print(f"{_tag(worker_id, 'STOP')} SparkSession stopped.")
    except Exception:
        pass
