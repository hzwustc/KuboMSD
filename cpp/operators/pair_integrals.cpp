#include "pair_integrals.hpp"
#include "radial_tables.hpp"
#include "angular.hpp"
#include <omp.h>
#include <set>
#include <exception>
namespace kubo {
Blocks evaluate_pairs(const Tables& t,const std::vector<Pair>& pairs,int threads) {
    require(threads>=1,"threads must be positive");Blocks out;out.offsets.push_back(0);std::set<I> seen;
    for(auto &p:pairs){
        require(p.id>=0 && seen.insert(p.id).second,"pair IDs must be nonnegative and unique");
        require(p.a>=0 && p.a<t.nspecies && p.b>=0 && p.b<t.nspecies,"species index out of range");
        require(std::isfinite(p.d[0]) && std::isfinite(p.d[1]) && std::isfinite(p.d[2]),"nonfinite displacement");
        auto &plan=t.plans[p.a*t.nspecies+p.b];I n=checked_product(plan.ni,plan.nj);
        require(out.offsets.back()<=INT64_MAX-n,"block offset overflow");out.offsets.push_back(out.offsets.back()+n);out.ids.push_back(p.id);
    }
    I total=out.offsets.back();
    require(checked_product(total,32)+checked_product(I(t.channels.size())*threads,8)<t.settings.budget,"pair workspace exceeds memory budget");
    out.s.resize(total);out.d.resize(checked_product(total,3));std::exception_ptr error;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(I i=0;i<I(pairs.size());i++)try{
        auto &p=pairs[i];auto &plan=t.plans[p.a*t.nspecies+p.b];
        std::array<double,3>d{p.d[0]/bohr,p.d[1]/bohr,p.d[2]/bohr};double r=std::hypot(std::hypot(d[0],d[1]),d[2]);
        if(r>=plan.cutoff)continue;
        auto angular=harmonics(7,d);std::vector<double> radial(t.channels.size(),NAN);
        for(auto &term:plan.terms){
            double &v=radial[term.channel];if(std::isnan(v))v=interpolate(t,t.channels[term.channel],r);
            double value=term.coefficient*angular[term.harmonic]*v;I idx=out.offsets[i]+term.out;
            if(term.component==0)out.s[idx]+=value;else out.d[(term.component-1)*total+idx]+=value;
        }
    }catch(...){
        #pragma omp critical
        {if(!error)error=std::current_exception();}
    }
    if(error)std::rethrow_exception(error);return out;
}
}
