"""Portable native stochastic kernels; build locally on each target machine."""
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    ext_modules=[
        Pybind11Extension(
            "kubo_msd._native",
            ["cpp/native.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
        Pybind11Extension(
            "kubo_msd._operators_native",
            ["cpp/operators/bindings.cpp", "cpp/operators/angular.cpp",
             "cpp/operators/radial_tables.cpp", "cpp/operators/pair_integrals.cpp",
             "cpp/operators/neighbors.cpp", "cpp/operators/assembly.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
