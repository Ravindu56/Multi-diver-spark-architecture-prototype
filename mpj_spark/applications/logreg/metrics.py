# =============================================================================
# mpj_spark/applications/logreg/metrics.py
# Phase 3 — Issue #9 — Step 6b: Per-Epoch Metrics Collection
#
# PURPOSE
# -------
# LogRegMetricsCollector is the logreg equivalent of KMeansMetricsCollector
# from kmeans/metrics.py.  It records per-epoch timing and convergence
# signals and writes per-rank CSV/JSON files plus an aggregated CSV across
# all ranks.
#
# PER-EPOCH FIELDS (record_epoch)
# --------------------------------
#   epoch         int    — current epoch number (1-indexed)
#   spark_time_s  float  — seconds inside compute_gradient_spark() only
#   sync_time_s   float  — seconds for the full MPI-collective window
#                           (allreduce_gradients + check_loss_convergence)
#   epoch_time_s  float  — wall-clock seconds for the full epoch
#   grad_norm     float  — L2 norm of the global gradient after Allreduce
#   global_loss   float  — cross-entropy loss averaged across all ranks
#   weight_norm   float  — L2 norm of the weight vector after update
#
# RUN-LEVEL FIELDS (record_run)
# ------------------------------
#   total_time_s    float  — wall-clock from first scatter to spark.stop()
#   epochs_run      int    — number of epochs actually completed
#   converged       bool   — did the run meet the loss-delta tolerance?
#   dataset_size    int    — total rows across all ranks
#   num_ranks       int    — MPI world size
#   learning_rate   float  — value used
#   tol             float  — convergence tolerance used
#
# OUTPUT FILES (written by each rank independently)
# --------------------------------------------------
#   {output_dir}/logreg_rank{rank}_epochs.csv   — per-epoch rows
#   {output_dir}/logreg_rank{rank}_run.json     — run-level summary
#
# AGGREGATED OUTPUT (written by rank 0 only)
# ------------------------------------------
#   {output_dir}/logreg_all_ranks_epochs.csv
#     — all per-rank epoch CSVs concatenated with a "rank" column
# =============================================================================

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LogRegMetricsCollector:
    """Collect and persist per-epoch metrics for the LogReg Allreduce runner."""

    # Ordered column names for the per-epoch CSV
    EPOCH_FIELDS: list[str] = [
        "epoch",
        "spark_time_s",
        "sync_time_s",
        "epoch_time_s",
        "grad_norm",
        "global_loss",
        "weight_norm",
    ]

    def __init__(self, rank: int, output_dir: str = "./metrics") -> None:
        self.rank = rank
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._epochs: list[dict[str, Any]] = []
        self._run: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Per-epoch recording
    # ------------------------------------------------------------------

    def record_epoch(
        self,
        epoch: int,
        spark_time_s: float,
        sync_time_s: float,
        epoch_time_s: float,
        grad_norm: float,
        global_loss: float,
        weight_norm: float,
    ) -> None:
        """Append one epoch's metrics to the in-memory list."""
        self._epochs.append(
            {
                "epoch": epoch,
                "spark_time_s": round(spark_time_s, 6),
                "sync_time_s": round(sync_time_s, 6),
                "epoch_time_s": round(epoch_time_s, 6),
                "grad_norm": round(float(grad_norm), 8),
                "global_loss": round(float(global_loss), 8),
                "weight_norm": round(float(weight_norm), 8),
            }
        )

    # ------------------------------------------------------------------
    # Run-level recording
    # ------------------------------------------------------------------

    def record_run(
        self,
        total_time_s: float,
        epochs_run: int,
        converged: bool,
        dataset_size: int,
        num_ranks: int,
        learning_rate: float,
        tol: float,
    ) -> None:
        """Store run-level summary (called once after the epoch loop)."""
        self._run = {
            "rank": self.rank,
            "total_time_s": round(total_time_s, 4),
            "epochs_run": epochs_run,
            "converged": converged,
            "dataset_size": dataset_size,
            "num_ranks": num_ranks,
            "learning_rate": learning_rate,
            "tol": tol,
        }

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def to_csv(self) -> str:
        """Write per-epoch rows to {output_dir}/logreg_rank{rank}_epochs.csv."""
        path = os.path.join(self.output_dir, f"logreg_rank{self.rank}_epochs.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.EPOCH_FIELDS)
            writer.writeheader()
            writer.writerows(self._epochs)
        logger.info("[rank %d] Epochs CSV written: %s", self.rank, path)
        return path

    def to_json(self) -> str:
        """Write run-level summary to {output_dir}/logreg_rank{rank}_run.json."""
        path = os.path.join(self.output_dir, f"logreg_rank{self.rank}_run.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._run, fh, indent=2)
        logger.info("[rank %d] Run JSON written: %s", self.rank, path)
        return path

    def summary_table(self) -> list[dict[str, Any]]:
        """Return the list of per-epoch dicts (used by the CLI summary printer)."""
        return list(self._epochs)

    # ------------------------------------------------------------------
    # Cross-rank aggregation (rank 0 only)
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_across_ranks(
        output_dir: str,
        num_ranks: int,
        output_filename: str = "logreg_all_ranks_epochs.csv",
    ) -> str | None:
        """
        Concatenate per-rank epoch CSVs into a single aggregated file.
        Adds a "rank" column so each row can be traced back to its origin.
        Called by rank 0 after all ranks have written their individual files.

        Parameters
        ----------
        output_dir      : directory containing rank CSV files
        num_ranks       : MPI world size
        output_filename : name of the aggregated output file

        Returns
        -------
        Path to the aggregated CSV, or None if no rank files were found.
        """
        all_rows: list[dict[str, Any]] = []
        for r in range(num_ranks):
            path = os.path.join(output_dir, f"logreg_rank{r}_epochs.csv")
            if not os.path.exists(path):
                logger.warning("Aggregation: rank %d CSV not found at %s", r, path)
                continue
            with open(path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    row["rank"] = str(r)
                    all_rows.append(row)

        if not all_rows:
            logger.warning("Aggregation: no rank CSV files found — skipping.")
            return None

        # Write aggregated CSV with "rank" as first column
        out_path = os.path.join(output_dir, output_filename)
        fieldnames = ["rank"] + LogRegMetricsCollector.EPOCH_FIELDS
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        logger.info("[rank 0] Aggregated epochs CSV written: %s", out_path)
        return out_path
