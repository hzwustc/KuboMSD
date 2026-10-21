#pragma once
#include "types.hpp"
#include <complex>
namespace kubo {
using C=std::complex<double>;
struct CSR {std::vector<I> indices,indptr;std::vector<C> data;};
CSR csr(const I*row,const I*col,const C*data,I count,I n,I budget);
}
