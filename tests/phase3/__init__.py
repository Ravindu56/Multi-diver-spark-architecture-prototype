# tests/phase3 — Phase 3 MPI integration tests
# All tests in this package that require a real MPI environment
# are guarded with @pytest.mark.skipif(mpi_size < 2, ...) and are
# skipped automatically in single-rank CI.  Run manually with:
#   mpirun --oversubscribe -n 5 pytest tests/phase3/ -v
