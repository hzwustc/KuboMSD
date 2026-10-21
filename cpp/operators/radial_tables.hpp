#pragma once
#include "types.hpp"
namespace kubo {
Tables build_tables(std::vector<Radial> radial,int nspecies,Settings settings);
void make_plans(Tables& t);
double interpolate(const Tables& t,const Channel& c,double r);
}
