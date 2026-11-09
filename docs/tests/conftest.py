"""Pin BLAS to one thread before numpy is imported: on small shared machines OpenBLAS thread
oversubscription made 400x400 dense SVDs ~500x slower (seen in S001)."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
