import pytest

@pytest.fixture
def gpu(request):
    from kubo_msd.backends.gpu import GPUBackend
    from kubo_msd.errors import BackendUnavailable
    try: return GPUBackend()
    except BackendUnavailable as exc:
        if request.config.getoption('--require-gpu'): pytest.fail(str(exc),pytrace=False)
        pytest.skip('GPU numerical acceptance NOT RUN: '+str(exc))
