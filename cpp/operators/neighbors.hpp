#pragma once
#include "types.hpp"
namespace kubo {
struct Neighbors {std::vector<I> keys;std::vector<double> displacement;};
Neighbors neighbors(const double* positions,const double* cell,const double* inverse,const double* cut,I n,I budget);
}
