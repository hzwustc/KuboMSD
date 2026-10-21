def test_native_operator_capabilities_distinguish_ready():
    from kubo_msd import _native, _operators_native
    c=_operators_native.capabilities()
    assert c['openmp'] and c['compiler']
    assert isinstance(c['fast_math_available'],bool)
    assert hasattr(_native,'build_info')
