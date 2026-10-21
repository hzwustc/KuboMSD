#pragma once
#include "types.hpp"
namespace kubo {
std::pair<std::vector<double>,std::vector<double>> gauss(int n);
std::vector<double> harmonics(int maxl, const std::array<double,3>& d);
double gaunt(int la,int ma,int lb,int mb,int lc,int mc);
void bessel(int maxl,double x,double* out);
}
