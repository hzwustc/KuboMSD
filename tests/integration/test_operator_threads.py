from test_fast_operator_assembly import fixture
from numpy.testing import assert_allclose

def test_threads_preserve_all_matrices_and_restore_blas():
    from kubo_msd.operators.builder import BuilderOptions,build_operators
    from threadpoolctl import threadpool_info
    s,o,h=fixture(); before=threadpool_info()
    a=build_operators(s,o,h,BuilderOptions(operator_builder='cpp',threads=1))
    for threads in (2,4):
        b=build_operators(s,o,h,BuilderOptions(operator_builder='cpp-omp',threads=threads))
        assert_allclose(a.S.data,b.S.data,rtol=1e-12,atol=1e-13)
        for q in 'xyz':
            for x,y in zip(a.composite[q],b.composite[q]):assert_allclose(x.toarray(),y.toarray(),rtol=1e-12,atol=1e-13)
    assert [v['num_threads'] for v in before]==[v['num_threads'] for v in threadpool_info()]
