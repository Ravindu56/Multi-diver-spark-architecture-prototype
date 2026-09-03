# tests/logreg/test_allreduce.py
#
# Unit tests for:
#   mpj_spark.applications.logreg.allreduce.allreduce_gradients()
#   mpj_spark.applications.logreg.allreduce.check_loss_convergence()
#
# MPI strategy:
#   comm is a MagicMock whose Allreduce / bcast side_effects simulate the
#   MPI collective operations without requiring mpi4py or OpenMPI.
#   This lets the tests run on any machine, including Windows laptops
#   where mpi4py is not installed.
#
# NO Spark is needed for these tests — allreduce.py has zero PySpark imports
# at the top level; Spark only enters through run_logreg_allreduce().
#
# Update rule under test (since 5ea5c59; the cosine-decay LR schedule is
# applied by the caller, L2 weight decay inside allreduce_gradients):
#   w_new = w - lr * (grad_avg + reg_param * w)
# reg_param=0 cases assert pure SGD; reg_param>0 cases assert L2 decay.


from unittest.mock import MagicMock

import numpy as np
import pytest

from mpj_spark.applications.logreg.allreduce import allreduce_gradients

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comm_allreduce(num_ranks: int):
    """
    Return a MagicMock comm whose Allreduce copies
    sendbuf * num_ranks into recvbuf, simulating MPI.SUM over num_ranks ranks.
    Buffer-level protocol: args are ([send, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM).
    """

    def _allreduce(sendbuf_pair, recvbuf_pair, op=None):
        send_arr = sendbuf_pair[0]
        recv_arr = recvbuf_pair[0]
        recv_arr[:] = send_arr * num_ranks

    comm = MagicMock()
    comm.Allreduce.side_effect = _allreduce
    return comm


# ---------------------------------------------------------------------------
# allreduce_gradients — weight update correctness
# ---------------------------------------------------------------------------


class TestAllreduceGradients:
    def test_weight_update_single_rank(self):
        """With 1 rank, global_grad == grad_local and w_new = w - lr * grad (pure SGD)."""
        comm = _make_comm_allreduce(num_ranks=1)
        w = np.zeros(4, dtype=np.float64)
        grad_local = np.array([1.0, 2.0, 3.0, 4.0])
        lr = 0.1

        w_new, global_grad = allreduce_gradients(
            comm, size=1, w=w, grad_local=grad_local, learning_rate=lr, reg_param=0.0
        )

        np.testing.assert_allclose(global_grad, grad_local)
        np.testing.assert_allclose(w_new, -lr * grad_local)

    @pytest.mark.parametrize("reg_param", [0.0, 0.01], ids=["pure_sgd", "l2_decay"])
    def test_weight_update_three_ranks(self, reg_param):
        """
        With 3 ranks the SUM is grad_local * 3.
        After dividing by size=3, global_grad == grad_local.
        Post-5ea5c59 rule: w_new = w - lr * (global_grad + reg_param * w);
        reg_param=0.0 reduces to the original pure-SGD expectation.
        """
        comm = _make_comm_allreduce(num_ranks=3)
        w = np.ones(3, dtype=np.float64)
        grad_local = np.array([0.5, 1.0, 1.5])
        lr = 0.01

        w_new, global_grad = allreduce_gradients(
            comm,
            size=3,
            w=w,
            grad_local=grad_local,
            learning_rate=lr,
            reg_param=reg_param,
        )

        np.testing.assert_allclose(global_grad, grad_local, rtol=1e-10)
        expected_w = w - lr * (global_grad + reg_param * w)
        np.testing.assert_allclose(w_new, expected_w, rtol=1e-10)

    def test_allreduce_called_once(self):
        """comm.Allreduce must be called exactly once per weight-update step."""
        comm = _make_comm_allreduce(num_ranks=2)
        w = np.zeros(2)
        grad_local = np.array([0.1, 0.2])

        allreduce_gradients(comm, size=2, w=w, grad_local=grad_local, learning_rate=0.05)

        assert comm.Allreduce.call_count == 1

    def test_output_shape_preserved(self):
        """w_new and global_grad must have the same shape as grad_local."""
        D = 10
        comm = _make_comm_allreduce(num_ranks=2)
        w = np.zeros(D)
        grad_local = np.random.default_rng(0).uniform(size=D)

        w_new, global_grad = allreduce_gradients(
            comm, size=2, w=w, grad_local=grad_local, learning_rate=0.01
        )

        assert w_new.shape == (D,)
        assert global_grad.shape == (D,)

    def test_zero_gradient_no_weight_change(self):
        """Pure SGD (reg_param=0): a zero local gradient leaves w unchanged."""
        comm = _make_comm_allreduce(num_ranks=2)
        w = np.array([1.0, -2.0, 3.0])
        grad_local = np.zeros(3)

        w_new, _ = allreduce_gradients(
            comm, size=2, w=w, grad_local=grad_local, learning_rate=0.1, reg_param=0.0
        )

        np.testing.assert_array_equal(w_new, w)

    def test_zero_gradient_l2_weight_decay(self):
        """
        Zero gradient with default reg_param=0.01: only the L2 decay term acts.
        w_new = w - lr * reg_param * w = w * (1 - lr * reg_param).
        Issue #67 fingerprint: lr=0.1, reg=0.01 -> [0.999, -1.998, 2.997].
        """
        comm = _make_comm_allreduce(num_ranks=2)
        w = np.array([1.0, -2.0, 3.0])
        grad_local = np.zeros(3)
        lr = 0.1

        w_new, _ = allreduce_gradients(comm, size=2, w=w, grad_local=grad_local, learning_rate=lr)

        np.testing.assert_allclose(w_new, w * (1.0 - lr * 0.01), rtol=1e-12)
        np.testing.assert_allclose(w_new, [0.999, -1.998, 2.997], rtol=1e-12)

    def test_default_reg_param_is_0_01(self):
        """Omitting reg_param must behave identically to reg_param=0.01."""
        w = np.array([2.0, -4.0])
        grad_local = np.array([0.5, 0.5])

        comm_default = _make_comm_allreduce(num_ranks=1)
        w_default, _ = allreduce_gradients(
            comm_default, size=1, w=w, grad_local=grad_local, learning_rate=0.1
        )
        comm_explicit = _make_comm_allreduce(num_ranks=1)
        w_explicit, _ = allreduce_gradients(
            comm_explicit,
            size=1,
            w=w,
            grad_local=grad_local,
            learning_rate=0.1,
            reg_param=0.01,
        )

        np.testing.assert_allclose(w_default, w_explicit, rtol=1e-12)

    def test_large_learning_rate_diverges_from_zero(self):
        """With a very large lr and positive gradient, weights should decrease."""
        comm = _make_comm_allreduce(num_ranks=1)
        w = np.array([5.0, 5.0])
        grad_local = np.array([1.0, 1.0])

        w_new, _ = allreduce_gradients(
            comm, size=1, w=w, grad_local=grad_local, learning_rate=10.0, reg_param=0.0
        )

        assert np.all(w_new < w)


# ---------------------------------------------------------------------------
# check_loss_convergence — tested through its pure-logic parts
# ---------------------------------------------------------------------------
# check_loss_convergence() mixes MPI + Spark + convergence logic in one
# function.  We test the convergence arithmetic directly via a thin wrapper
# that isolates the delta < tol check, and separately verify the Allreduce
# and bcast call patterns on the mock comm.


class TestCheckLossConvergenceMath:
    """
    Verify convergence logic by calling check_loss_convergence() with a
    mock RDD and mock comm, simulating single-rank execution (rank=0, size=1).
    """

    def _make_mock_rdd(self, rows):
        """
        Build a MagicMock RDD that supports .map().reduce() and .count().
        rows: list of (np.ndarray, float)
        """
        mock_rdd = MagicMock()

        def _map(fn):
            mapped = MagicMock()
            results = [fn(r) for r in rows]
            mapped.reduce.side_effect = lambda f: __import__("functools").reduce(f, results)
            return mapped

        mock_rdd.map.side_effect = _map
        mock_rdd.count.return_value = len(rows)
        return mock_rdd

    def _make_comm_rank0(self, num_ranks=1):
        """Comm mock that simulates rank-0 Allreduce + bcast for size=num_ranks."""

        def _allreduce(send_pair, recv_pair, op=None):
            recv_pair[0][:] = send_pair[0] * num_ranks

        comm = MagicMock()
        comm.Allreduce.side_effect = _allreduce
        # bcast returns whatever rank 0 computes (converged flag)
        comm.bcast.side_effect = lambda val, root=0: val
        return comm

    def test_converged_when_delta_below_tol(self):
        """
        If abs(prev_loss - global_loss) < tol the function must return True.
        We engineer a dataset where the loss is ~0.693 (random 50/50 labels,
        zero weights) and set prev_loss just above it so delta < tol.
        """
        from mpj_spark.applications.logreg.allreduce import check_loss_convergence

        # Two rows: label=1 and label=0 with the same feature vector
        # Weights are zero so sigmoid = 0.5 for both
        # loss per sample = -log(0.5) = 0.6931...
        rows = [
            (np.array([1.0, 0.0]), 1.0),
            (np.array([1.0, 0.0]), 0.0),
        ]
        mock_rdd = self._make_mock_rdd(rows)
        comm = self._make_comm_rank0(num_ranks=1)
        w = np.zeros(2)
        # Expected global_loss ≈ 0.6931472
        # Set prev_loss = global_loss + 5e-5 so delta = 5e-5 < tol=1e-4
        expected_loss = -np.log(0.5)
        prev_loss = expected_loss + 5e-5

        converged, global_loss = check_loss_convergence(
            comm=comm,
            rank=0,
            size=1,
            data_rdd=mock_rdd,
            w=w,
            prev_loss=prev_loss,
            tol=1e-4,
            epoch=1,
        )

        assert converged is True
        assert abs(global_loss - expected_loss) < 1e-6

    def test_not_converged_when_delta_above_tol(self):
        """If delta >= tol, converged must be False."""
        from mpj_spark.applications.logreg.allreduce import check_loss_convergence

        rows = [
            (np.array([1.0, 0.0]), 1.0),
            (np.array([1.0, 0.0]), 0.0),
        ]
        mock_rdd = self._make_mock_rdd(rows)
        comm = self._make_comm_rank0(num_ranks=1)
        w = np.zeros(2)
        # Set prev_loss = 0.0 so delta = ~0.693 >> tol=1e-4
        converged, _ = check_loss_convergence(
            comm=comm,
            rank=0,
            size=1,
            data_rdd=mock_rdd,
            w=w,
            prev_loss=0.0,
            tol=1e-4,
            epoch=1,
        )

        assert converged is False

    def test_bcast_called_once(self):
        """comm.bcast must be called exactly once to propagate the stop flag."""
        from mpj_spark.applications.logreg.allreduce import check_loss_convergence

        rows = [(np.array([1.0]), 1.0)]
        mock_rdd = self._make_mock_rdd(rows)
        comm = self._make_comm_rank0()
        w = np.zeros(1)

        check_loss_convergence(
            comm=comm,
            rank=0,
            size=1,
            data_rdd=mock_rdd,
            w=w,
            prev_loss=float("inf"),
            tol=1e-4,
            epoch=1,
        )

        assert comm.bcast.call_count == 1
