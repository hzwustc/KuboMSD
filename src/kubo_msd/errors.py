class KuboMSDError(ValueError):
    """Invalid input or failed numerical acceptance; never silently repaired."""

class BackendUnavailable(KuboMSDError):
    pass

class NumericalError(KuboMSDError):
    pass
