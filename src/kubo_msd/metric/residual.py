"""Column-local reliable updates; pure array operations, including on CUDA."""


def assess_residual(r_recursive, r_true, threshold, active, gap_ratio, *, xp):
    nr = xp.sqrt(xp.sum(xp.abs(r_recursive)**2, axis=0))
    nt = xp.sqrt(xp.sum(xp.abs(r_true)**2, axis=0))
    gap = xp.sqrt(xp.sum(xp.abs(r_true-r_recursive)**2, axis=0))
    converged = active & (nt <= threshold)
    restart = active & ~converged & (
        (gap > gap_ratio*xp.maximum(nt, threshold)) | (nr <= threshold))
    return converged, restart
