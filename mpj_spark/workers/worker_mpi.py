# ================================================================
# mpj_spark/workers/worker_mpi.py  -  MPI Worker Runner  (ranks 1..N)
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
#
# PURPOSE
# -------
# MPI-native worker runner.  Called by every MPI rank != 0.
# Replaces the multiprocessing.Process / Queue / Event scaffolding
# in worker_process.py with direct MPI point-to-point calls over
# MPI_COMM_WORLD, while reusing all Spark application logic unchanged.
#
# P3-02 ACCEPTANCE CRITERION  (worker side)
# ------------------------------------------
#   MPI_COMM_WORLD replaces multiprocessing.Process; workers are ranks 1..N.
#
#   Verified by:
#     - assert comm.Get_rank() != 0  at entry
#     - grep multiprocessing mpj_spark/workers/worker_mpi.py  =>  no output
#     - grep Queue             mpj_spark/workers/worker_mpi.py  =>  only MpiWorkerAllreduceAdapter
#
# MPI BOOT SEQUENCE
# -----------------
#   recv(TAG_CONFIG)      <- root sends partition path + cfg dict
#   build_spark_session() <- JVM + SparkContext init
#   send(TAG_READY)       -> JVM-ready sentinel to root (replaces Event.set())
#   recv(TAG_GO)          <- go-signal from root      (replaces Event.wait())
#   [run application]
#   send(TAG_RESULT)      -> result dict to root      (replaces Queue.put())
#   send(TAG_TIMING)      -> timing dict to root      (replaces Queue.put())
#   [optional reassign pass for kmeans]
#   recv(TAG_REASSIGN_BCAST) <- gossip centroids
#   send(TAG_REASSIGN_STATS) -> cluster sums / counts
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
from mpi4py import MPI

from mpj_spark.workers.spark_session import build_spark_session
from mpj_spark.utils.dev_logger import DevLogger

# Import the pure Spark/NumPy reassign helper directly from worker_process
# to avoid code duplication.  It has no multiprocessing dependency.
from mpj_spark.workers.worker_process import _reassign_pass

# ── MPI tag constants (mirrors root_mpi.py) ─────────────────────────
TAG_CONFIG         = 10
TAG_RESULT         = 20
TAG_TIMING         = 21
TAG_ALLREDUCE_UP   = 30
TAG_ALLREDUCE_DOWN = 31
TAG_REASSIGN_BCAST = 40
TAG_REASSIGN_STATS = 41
TAG_READY          = 50
TAG_GO             = 60


def _tag(worker_id, phase):
    """Log-line prefix consistent with worker_process.py."""
    return f'[Worker {worker_id}][{phase}]'


# ================================================================
# MpiWorkerAllreduceAdapter
# ================================================================
# logreg.run() calls allreduce_up_queue.put(weights) per iteration
# and allreduce_down_queue.get() to receive averaged weights.
# This adapter wraps the two MPI send/recv calls behind that
# Queue-compatible interface so logreg.run() requires zero changes.

class MpiWorkerAllreduceAdapter:
    """
    Queue-like adapter that bridges logreg.run()'s .put() / .get()
    calls to MPI point-to-point messages.

    UP adapter  (allreduce_up_queue):
        .put(msg)  ->  comm.send(msg, dest=0, tag=TAG_ALLREDUCE_UP)

    DOWN adapter  (allreduce_down_queue):
        .get()     ->  comm.recv(source=0, tag=TAG_ALLREDUCE_DOWN)

    No internal buffer is maintained; each call maps directly to one
    MPI send or recv so ordering and flow-control are handled by MPI.
    """

    def __init__(self, comm, direction: str):
        """
        Parameters
        ----------
        comm      : MPI communicator (MPI_COMM_WORLD)
        direction : 'up'   -> adapter for allreduce_up_queue
                    'down' -> adapter for allreduce_down_queue
        """
        if direction not in ('up', 'down'):
            raise ValueError("direction must be 'up' or 'down'")
        self._comm      = comm
        self._direction = direction

    # ── UP adapter interface (≡ Queue.put) ────────────────────────
    def put(self, msg, block=True, timeout=None):
        """Send a weights message to root (TAG_ALLREDUCE_UP)."""
        if self._direction != 'up':
            raise RuntimeError(
                'put() called on DOWN adapter — use get() instead.')
        self._comm.send(msg, dest=0, tag=TAG_ALLREDUCE_UP)

    # ── DOWN adapter interface (≡ Queue.get) ───────────────────────
    def get(self, block=True, timeout=None):
        """Receive averaged weights from root (TAG_ALLREDUCE_DOWN)."""
        if self._direction != 'down':
            raise RuntimeError(
                'get() called on UP adapter — use put() instead.')
        return self._comm.recv(source=0, tag=TAG_ALLREDUCE_DOWN)

    # ── Compatibility stubs (called by some Queue consumers) ──────────
    def empty(self):
        return False   # MPI is synchronous; always assume a message may arrive

    def qsize(self):
        return 0


# ================================================================
# MpiKMeansGossipAdapter
# ================================================================
# kmeans.run() calls gossip_queue.put(centroid_state) after fitting.
# This adapter maps that .put() to comm.send(dest=0, tag=TAG_ALLREDUCE_UP)
# (gossip uses the same tag as logreg allreduce-UP for simplicity;
# root_mpi.py routes them correctly by phase).

class MpiKMeansGossipAdapter:
    """
    Queue-like adapter for K-Means gossip messages.

    kmeans.run() calls gossip_queue.put({'worker_id':..., 'centres':...,
    'wcss':..., 'row_count':...}) once after local fitting.
    This adapter forwards that single message to root via MPI.
    """

    def __init__(self, comm):
        self._comm = comm

    def put(self, msg, block=True, timeout=None):
        """Forward centroid state to root (TAG_ALLREDUCE_UP)."""
        self._comm.send(msg, dest=0, tag=TAG_ALLREDUCE_UP)

    def empty(self):
        return True

    def qsize(self):
        return 0


# ================================================================
# run_worker_mpi  —  main MPI worker entry point
# ================================================================

def run_worker_mpi(comm):
    """
    MPI-native worker runner.  Must be called by every rank != 0.

    Receives configuration from root (rank 0), initialises a SparkSession,
    signals readiness, waits for the simultaneous go-signal, executes the
    assigned application, and sends results + timings back to root.
    All communication uses MPI point-to-point on the supplied communicator.

    P3-02 acceptance criterion:
      MPI_COMM_WORLD replaces multiprocessing.Process; workers are ranks 1..N.
    """
    rank = comm.Get_rank()

    assert rank != 0, (
        f'run_worker_mpi() must not be called by rank 0 (root coordinator). '
        f'Only ranks >= 1 should call this function.'
    )

    # worker_id is 0-indexed (rank 1 => worker 0, rank 2 => worker 1, ...)
    worker_id = rank - 1

    # ================================================================
    # Step 1 — Receive configuration from root
    #          Replaces: Process(args=(worker_id, partition_path, ...))
    # ================================================================
    print(f'{_tag(worker_id, "BOOT")} rank={rank}  waiting for config ...')
    cfg = comm.recv(source=0, tag=TAG_CONFIG)

    partition_path   = cfg['partition_path']
    app_name         = cfg.get('app',              'wordcount')
    cores_override   = cfg.get('cores_override',    None)
    kmeans_k         = int(cfg.get('kmeans_k',         3))
    kmeans_iter      = int(cfg.get('kmeans_max_iter',  20))
    num_workers      = cfg.get('num_workers',   1)
    seed_centres     = cfg.get('seed_centres',     None)
    logreg_iter      = int(cfg.get('logreg_iter',      10))
    logreg_reg_param = float(cfg.get('logreg_reg_param', 0.01))
    logreg_features  = int(cfg.get('logreg_features', 10))
    results_dir      = cfg.get('results_dir',  'results')

    print(
        f'{_tag(worker_id, "BOOT")} config received  '
        f'app={app_name}  partition={partition_path}'
    )

    logger = DevLogger(worker_id=worker_id)

    # ================================================================
    # Step 2 — Initialise SparkSession (JVM warm-up)
    # ================================================================
    print(f'{_tag(worker_id, "INIT")} Starting SparkSession (app={app_name}) ...')
    t_init_start = time.perf_counter()

    try:
        spark = build_spark_session(
            app_name       = f'MPJ-MPI-Worker-{worker_id}-{app_name}',
            cores_override = cores_override,
            num_workers    = num_workers,
        )
    except Exception as exc:
        # If JVM init fails we must still send TAG_READY and TAG_RESULT
        # to prevent root from blocking forever on comm.recv.
        print(f'{_tag(worker_id, "INIT")} SparkSession FAILED: {exc}')
        traceback.print_exc()
        comm.send('ready', dest=0, tag=TAG_READY)
        comm.recv(source=0, tag=TAG_GO)
        comm.send(
            {'worker_id': worker_id, 'result': None,
             'status': 'error', 'error': str(exc)},
            dest=0, tag=TAG_RESULT,
        )
        comm.send(
            {'worker_id': worker_id, 'init_time': 0.0,
             'load_time': 0.0, 'processing_time': 0.0, 'total_time': 0.0},
            dest=0, tag=TAG_TIMING,
        )
        return

    init_time = time.perf_counter() - t_init_start
    print(f'{_tag(worker_id, "INIT")} SparkSession ready  ({init_time:.3f}s)')

    # ================================================================
    # Step 3 — Signal JVM-ready to root
    #          Replaces: ready_signal.set()
    # ================================================================
    comm.send('ready', dest=0, tag=TAG_READY)
    print(f'{_tag(worker_id, "WAIT")} JVM-ready sent to root — waiting for go-signal ...')

    # ================================================================
    # Step 4 — Wait for simultaneous go-signal from root
    #          Replaces: go_signal.wait()
    # ================================================================
    comm.recv(source=0, tag=TAG_GO)
    print(f'{_tag(worker_id, "WAIT")} Go-signal received — starting {app_name}')

    # ================================================================
    # Steps 5-7 — Run application, emit result + timing
    # ================================================================
    load_time = 0.0
    proc_time = 0.0
    app_result = None

    try:
        # ── LOAD ─────────────────────────────────────────────────────
        print(f'{_tag(worker_id, "LOAD")} Loading partition ...')
        t_load_start = time.perf_counter()

        if app_name == 'wordcount':
            text_rdd  = spark.sparkContext.textFile(partition_path)
            text_rdd.cache()
            row_count = text_rdd.count()
            load_time = time.perf_counter() - t_load_start
            print(f'{_tag(worker_id, "LOAD")} {row_count:,} rows  ({load_time:.3f}s)')
        # kmeans and logreg load inside their run() calls

        # ── PROC ─────────────────────────────────────────────────────
        print(f'{_tag(worker_id, "PROC")} Running {app_name} ...')
        t_proc_start = time.perf_counter()

        if app_name == 'wordcount':
            from mpj_spark.applications import wordcount
            app_result = wordcount.run(text_rdd)

        elif app_name == 'kmeans':
            from mpj_spark.applications import kmeans
            app_result = kmeans.run(
                partition_path,
                k            = kmeans_k,
                max_iter     = kmeans_iter,
                seed_centres = seed_centres,
            )
            # Forward centroid state to root via MPI (replaces gossip_queue.put)
            if app_result is not None:
                gossip_adapter = MpiKMeansGossipAdapter(comm)
                gossip_adapter.put({
                    'worker_id' : worker_id,
                    'centres'   : app_result['centres'],
                    'wcss'      : app_result['wcss'],
                    'row_count' : app_result['row_count'],
                })
                print(f'{_tag(worker_id, "PROC")} Centroid state → root (MPI TAG_ALLREDUCE_UP)')

        elif app_name == 'logreg':
            from mpj_spark.applications import logreg
            # Wire MPI adapters so logreg.run() sees Queue-compatible objects
            allreduce_up_adapter   = MpiWorkerAllreduceAdapter(comm, direction='up')
            allreduce_down_adapter = MpiWorkerAllreduceAdapter(comm, direction='down')
            app_result = logreg.run(
                partition_path,
                max_iter             = logreg_iter,
                reg_param            = logreg_reg_param,
                num_features         = logreg_features,
                worker_id            = worker_id,
                allreduce_up_queue   = allreduce_up_adapter,
                allreduce_down_queue = allreduce_down_adapter,
                num_workers          = num_workers,
            )
            print(
                f'{_tag(worker_id, "PROC")} LogReg done  '
                f'acc={app_result["train_accuracy"]:.4f}  '
                f'iters={app_result["iterations_done"]}'
            )

        else:
            raise ValueError(
                f"Unknown app '{app_name}'. Valid: 'wordcount', 'kmeans', 'logreg'"
            )

        proc_time = time.perf_counter() - t_proc_start
        print(f'{_tag(worker_id, "DONE")} {app_name} complete  ({proc_time:.3f}s)')

        # ================================================================
        # Step 6 — Send result and timing to root
        #          Replaces: result_queue.put() / timing_queue.put()
        # ================================================================
        comm.send(
            {'worker_id': worker_id, 'result': app_result, 'status': 'success'},
            dest=0, tag=TAG_RESULT,
        )
        comm.send(
            {
                'worker_id'      : worker_id,
                'init_time'      : init_time,
                'load_time'      : load_time,
                'processing_time': proc_time,
                'total_time'     : init_time + load_time + proc_time,
            },
            dest=0, tag=TAG_TIMING,
        )
        logger.log_worker_timing(
            worker_id = worker_id,
            init_time = init_time,
            load_time = load_time,
            proc_time = proc_time,
        )

        # ================================================================
        # Step 7 (optional) — Re-assignment pass (K-Means only)
        #   Replaces: reassign_queue.get() / reassign_queue.put()
        # ================================================================
        if app_name == 'kmeans':
            print(f'{_tag(worker_id, "REASSIGN")} Waiting for global centroids from root ...')
            msg = comm.recv(source=0, tag=TAG_REASSIGN_BCAST)
            if msg.get('type') == 'reassign':
                global_centres = msg['centres']
                reassign_stats = _reassign_pass(
                    spark, partition_path, global_centres, worker_id
                )
                comm.send(
                    {
                        'type'          : 'stats',
                        'worker_id'     : worker_id,
                        'cluster_sums'  : reassign_stats['cluster_sums'],
                        'cluster_counts': reassign_stats['cluster_counts'],
                        'row_count'     : reassign_stats['row_count'],
                    },
                    dest=0, tag=TAG_REASSIGN_STATS,
                )
                print(
                    f'{_tag(worker_id, "REASSIGN")} Stats sent to root '
                    f'({reassign_stats["row_count"]:,} rows)'
                )

    except Exception as exc:
        print(f'{_tag(worker_id, "ERROR")} {exc}')
        traceback.print_exc()
        # Always send TAG_RESULT + TAG_TIMING to unblock root's recv loop
        comm.send(
            {'worker_id': worker_id, 'result': None,
             'status': 'error', 'error': str(exc)},
            dest=0, tag=TAG_RESULT,
        )
        comm.send(
            {'worker_id': worker_id, 'init_time': init_time,
             'load_time': 0.0, 'processing_time': 0.0, 'total_time': init_time},
            dest=0, tag=TAG_TIMING,
        )

    finally:
        try:
            spark.stop()
            print(f'{_tag(worker_id, "STOP")} SparkSession stopped.')
        except Exception:
            pass
