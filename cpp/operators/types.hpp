#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <stdexcept>
namespace kubo {
constexpr double pi=3.141592653589793238462643383279502884;
constexpr double bohr=0.529177210903;
using I=std::int64_t;
struct Radial { int l, species; std::vector<double> radius, coeff; };
struct Settings { double kmax=32., dk=.1, step=.05; int order=4; I budget=2147483648LL; };
struct Channel { int a,b,lambda,power,L; std::vector<double> values; };
struct Term { I out; int channel, harmonic, component; double coefficient; };
struct PairPlan { int ni=0,nj=0; double cutoff=0; std::vector<Term> terms; };
struct Tables {
    Settings settings;
    std::vector<Radial> orbitals;
    std::vector<Channel> channels;
    std::vector<PairPlan> plans;
    int nspecies=0, ndistance=0;
    double step=0;
};
struct Pair { I id; int a,b; std::array<double,3> d; };
struct Blocks { std::vector<I> ids, offsets; std::vector<double> s,d; };
inline void require(bool ok,const std::string &message) { if(!ok) throw std::invalid_argument(message); }
inline I checked_product(I a,I b) {
    require(a>=0 && b>=0 && (b==0 || a<=INT64_MAX/b),"integer index/size overflow"); return a*b;
}
}
