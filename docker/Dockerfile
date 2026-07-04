# =============================================================================
# Dockerfile — MPJ-SPARK Multi-Driver Architecture (Phase 4)
# University of Jaffna — EC6070 — 2022/E/033 & 2022/E/090
#
# One container = one MPI rank = one independent Spark driver
#
# Build:
#   docker build -t mpj-spark:latest -f docker/Dockerfile .
#
# Usage:
#   See docker/docker-compose.yml and docker/entrypoint.sh
# =============================================================================

FROM python:3.11-slim-bookworm

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL maintainer="EC6070 Team — 2022E033 & 2022E090"
LABEL description="MPJ-SPARK Phase 4: mpi4py + OpenMPI + PySpark worker container"
LABEL phase="4"

# ---------------------------------------------------------------------------
# 1. OS-level dependencies
#    - openjdk-17-jdk  : required by PySpark (Spark 3.x targets JDK 17)
#    - libopenmpi-dev  : OpenMPI C headers + runtime (mpi4py links against this)
#    - openmpi-bin     : mpirun, mpiexec executables
#    - openssh-server  : needed so mpirun can spawn remote ranks via SSH
#    - openssh-client  : needed so rank 0 can SSH into worker containers
#    - rsync           : used by some MPI launchers for binary distribution
#    - procps          : ps, kill — useful for Spark process management
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jdk \
        libopenmpi-dev \
        openmpi-bin \
        openssh-server \
        openssh-client \
        rsync \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Java environment
#    JAVA_HOME must be set explicitly for PySpark to find the JVM.
#    The exact path depends on the JDK package installed above.
# ---------------------------------------------------------------------------
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ---------------------------------------------------------------------------
# 3. JVM module-access flags
#    Copied directly from mpj_spark_mpi.py os.environ.setdefault() block.
#    Setting here ensures they apply even when Spark is launched as a
#    subprocess from worker_process.py (not just from the MPI entry point).
# ---------------------------------------------------------------------------
ENV JAVA_TOOL_OPTIONS="--add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
--add-opens=java.base/java.nio=ALL-UNNAMED \
--add-opens=java.base/java.lang=ALL-UNNAMED \
--add-opens=java.base/java.util=ALL-UNNAMED \
-Djava.security.manager=allow"

# ---------------------------------------------------------------------------
# 4. PySpark Python pointers
#    Ensures Spark executors and drivers use the same Python binary as the
#    MPI process, preventing interpreter mismatch errors.
# ---------------------------------------------------------------------------
ENV PYSPARK_PYTHON=/usr/local/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3

# ---------------------------------------------------------------------------
# 5. MPI environment
#    OMPI_ALLOW_RUN_AS_ROOT / _CONFIRM: required when running inside Docker
#    containers where the process is often UID 0 (root).
#    OMPI_MCA_btl_vader_single_copy_mechanism: disables CMA (copy-on-write
#    memory) which causes hangs inside Docker without --privileged.
# ---------------------------------------------------------------------------
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# ---------------------------------------------------------------------------
# 6. Python dependencies
#    Copy requirements.txt first (layer-cache friendly — only rebuilds if
#    requirements change, not on every source code edit).
# ---------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# 7. Project source
#    Copy full project into /app so all modules are importable.
#    .dockerignore (to be added) should exclude __pycache__, .git, results/
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# 8. SSH setup for mpirun inter-container communication (P4-04)
#    - Generate a host RSA key for the SSH server
#    - Create /run/sshd (required by sshd -D)
#    - Generate a passwordless keypair for the mpirun user
#    - Authorise the public key so rank 0 can SSH into worker containers
#    - Disable StrictHostKeyChecking so mpirun does not hang on first connect
# ---------------------------------------------------------------------------
RUN mkdir -p /var/run/sshd \
    && ssh-keygen -A \
    && mkdir -p /root/.ssh \
    && ssh-keygen -t rsa -b 4096 -N "" -f /root/.ssh/id_rsa \
    && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys \
    && echo "Host *" >> /root/.ssh/config \
    && echo "    StrictHostKeyChecking no" >> /root/.ssh/config \
    && echo "    UserKnownHostsFile /dev/null" >> /root/.ssh/config \
    && chmod 600 /root/.ssh/config

# ---------------------------------------------------------------------------
# 9. Shared data directory
#    /data is the NFS mount point (P4-02). Creating it here ensures the
#    directory exists in the image; the actual NFS volume is mounted at
#    runtime via docker-compose.
# ---------------------------------------------------------------------------
RUN mkdir -p /data/input /data/output /data/results

# ---------------------------------------------------------------------------
# 10. Entrypoint
#     The entrypoint script (P4-04) starts sshd, generates the MPI hostfile,
#     and dispatches to root_main() or worker_main() based on MPI_RANK.
# ---------------------------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---------------------------------------------------------------------------
# Ports
#   22   — SSH (mpirun inter-container)
#   4040 — Spark Web UI (per driver; mapped per-container in compose)
# ---------------------------------------------------------------------------
EXPOSE 22 4040

ENTRYPOINT ["/entrypoint.sh"]
