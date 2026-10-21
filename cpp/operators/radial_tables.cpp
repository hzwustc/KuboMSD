#include "radial_tables.hpp"
#include "angular.hpp"
#include <map>
#include <algorithm>
#include <tuple>
namespace kubo {
// At R=0 Parseval gives an exact radial-product rule. On each interval
// the input splines are cubics, so r^(2+power)*fa*fb has degree <=9.
// Six-point Gauss integration is exact up to floating-point roundoff.
double onsite_product(const Radial& a,const Radial& b,int power) {
    double cut=std::min(a.radius.back(),b.radius.back());
    std::vector<double> grid;
    for(double r:a.radius)if(r<=cut)grid.push_back(r);
    for(double r:b.radius)if(r<=cut)grid.push_back(r);
    std::sort(grid.begin(),grid.end());grid.erase(std::unique(grid.begin(),grid.end()),grid.end());
    auto g=gauss(6);long double sum=0;
    auto value=[](const Radial& f,double r,int i) {
        double x=r-f.radius[i];const double* c=&f.coeff[4*i];
        return ((c[0]*x+c[1])*x+c[2])*x+c[3];
    };
    for(size_t j=0;j+1<grid.size();j++){
        double lo=grid[j],h=grid[j+1]-lo,mid=lo+h/2;
        int ia=std::upper_bound(a.radius.begin(),a.radius.end(),mid)-a.radius.begin()-1;
        int ib=std::upper_bound(b.radius.begin(),b.radius.end(),mid)-b.radius.begin()-1;
        for(int q=0;q<6;q++){
            double r=lo+(g.first[q]+1)*h/2;
            sum+=g.second[q]*h/2*std::pow(r,2+power)*value(a,r,ia)*value(b,r,ib);
        }
    }
    return double(sum);
}

Tables build_tables(std::vector<Radial> radial,int nspecies,Settings settings) {
    require(settings.kmax>0 && settings.dk>0 && settings.step>0 && settings.order>=2 && settings.order<=16,"invalid table settings");
    require(std::isfinite(settings.kmax+settings.dk+settings.step),"nonfinite table settings");
    Tables t; t.settings=settings;t.orbitals=std::move(radial);t.nspecies=nspecies;
    double maxcut=0;
    for(auto &r:t.orbitals)maxcut=std::max(maxcut,r.radius.back());
    require(settings.budget>0 && settings.kmax/settings.dk<settings.budget/64. && 2*maxcut/settings.step<std::min<double>(INT32_MAX-1,settings.budget/64.),"table grid exceeds index/memory limit");
    I nk=I(std::ceil(settings.kmax/settings.dk));if(nk%2)nk++;nk++;
    double dk=settings.kmax/(nk-1);t.settings.dk=dk;
    t.ndistance=std::max(6,int(std::ceil(2*maxcut/settings.step))+1);
    t.step=2*maxcut/(t.ndistance-1);t.settings.step=t.step;
    I bsize=checked_product(checked_product(8,t.ndistance),nk);
    require(checked_product(bsize,8)<settings.budget/2,"table Bessel workspace exceeds memory budget");
    I nch=0;
    for(auto &a:t.orbitals)for(auto &b:t.orbitals){
        nch+=std::min(a.l,b.l)+1;
        nch+=std::min(a.l+1,b.l)+1;
        if(a.l>0)nch+=std::min(a.l-1,b.l)+1;
    }
    I persistent=checked_product(checked_product(nch,t.ndistance),8);
    require(persistent+bsize*8+I(t.orbitals.size())*3*nk*8<settings.budget/2,"table channels/workspace exceed total memory budget");
    std::vector<double> B(bsize);
    for(int ir=0;ir<t.ndistance;ir++)for(I k=0;k<nk;k++){
        double v[8];bessel(7,ir*t.step*k*dk,v);
        double w=(k==0 || k==nk-1?1:(k%2?4:2))*dk/3*(k*dk)*(k*dk);
        for(int L=0;L<8;L++)B[(L*t.ndistance+ir)*nk+k]=v[L]*w;
    }
    using Key=std::tuple<int,int,int>;
    std::map<Key,std::vector<double>> transforms;
    auto g=gauss(settings.order);
    for(int a=0;a<int(t.orbitals.size());a++){
        auto &rad=t.orbitals[a];int la=rad.l;
        std::vector<Key> keys{{a,la,0},{a,la+1,1}};if(la>0)keys.emplace_back(a,la-1,1);
        for(auto key:keys)transforms[key]=std::vector<double>(nk,0.);
        for(I k=0;k<nk;k++){
            double sums[3]={0,0,0};
            for(int ir=0;ir<int(rad.radius.size())-1;ir++){
                double lo=rad.radius[ir],h=rad.radius[ir+1]-lo;
                const double* c=&rad.coeff[4*ir];
                for(int q=0;q<settings.order;q++){
                    double x=(g.first[q]+1)*h/2,r=lo+x,f=((c[0]*x+c[1])*x+c[2])*x+c[3];
                    double weight=g.second[q]*h/2*r*r*f,v[8];bessel(la+1,k*dk*r,v);
                    sums[0]+=weight*v[la];sums[1]+=weight*r*v[la+1];if(la>0)sums[2]+=weight*r*v[la-1];
                }
            }
            for(int j=0;j<int(keys.size());j++)transforms[keys[j]][k]=sums[j];
        }
    }
    for(auto &entry:transforms){
        auto [a,lambda,power]=entry.first;
        for(int b=0;b<int(t.orbitals.size());b++){
            int lb=t.orbitals[b].l;
            auto &fb=transforms.at({b,lb,0});
            std::vector<double> prod(nk);
            for(I k=0;k<nk;k++)prod[k]=entry.second[k]*fb[k];
            for(int L=std::abs(lambda-lb);L<=lambda+lb;L+=2){
                Channel c{a,b,lambda,power,L,std::vector<double>(t.ndistance)};
                for(int ir=0;ir<t.ndistance;ir++){
                    const double *bv=&B[(L*t.ndistance+ir)*nk];double sum=0;
                    for(I k=0;k<nk;k++)sum+=prod[k]*bv[k];c.values[ir]=sum;
                }
                if(L==0)c.values[0]=pi/2*onsite_product(t.orbitals[a],t.orbitals[b],power);
                t.channels.push_back(std::move(c));
            }
        }
    }
    require(checked_product(I(t.channels.size())*t.ndistance,8)+bsize*8<settings.budget,"table channels exceed memory budget");
    make_plans(t);return t;
}
void make_plans(Tables& t) {
    t.plans.clear();t.plans.resize(t.nspecies*t.nspecies);
    std::vector<int> orbital_offset(t.orbitals.size());std::vector<int> sizes(t.nspecies,0);
    for(int a=0;a<int(t.orbitals.size());a++){auto &r=t.orbitals[a];orbital_offset[a]=sizes[r.species];sizes[r.species]+=2*r.l+1;}
    I term_bytes=0;
    auto append_term=[&](PairPlan &plan,Term term){
        require(term_bytes+I(sizeof(Term))<t.settings.budget/4,"angular plan exceeds memory budget");
        plan.terms.push_back(term);term_bytes+=sizeof(Term);
    };
    std::map<std::array<int,6>,double> gc;
    auto G=[&](int a,int m,int b,int n,int c,int p){std::array<int,6> key{a,m,b,n,c,p};auto it=gc.find(key);if(it!=gc.end())return it->second;return gc[key]=gaunt(a,m,b,n,c,p);};
    for(int ci=0;ci<int(t.channels.size());ci++){
        auto &c=t.channels[ci];auto &a=t.orbitals[c.a];auto &b=t.orbitals[c.b];
        auto &plan=t.plans[a.species*t.nspecies+b.species];plan.ni=sizes[a.species];plan.nj=sizes[b.species];
        plan.cutoff=std::max(plan.cutoff,a.radius.back()+b.radius.back());
        double phase=((c.lambda-b.l-c.L)/2)%2?-1.:1.;
        for(int ma=0;ma<2*a.l+1;ma++)for(int mb=0;mb<2*b.l+1;mb++){
            I out=(orbital_offset[c.a]+ma)*I(plan.nj)+orbital_offset[c.b]+mb;
            for(int M=0;M<2*c.L+1;M++){
                if(c.power==0){
                    double v=8*phase*G(a.l,ma,b.l,mb,c.L,M);
                    if(v!=0)append_term(plan,{out,ci,c.L*c.L+M,0,v});
                }else for(int q=0;q<3;q++){
                    int qm=q==0?1:(q==1?2:0);double sign=q==2?1.:-1.;double v=0;
                    for(int mu=0;mu<2*c.lambda+1;mu++)v+=G(a.l,ma,1,qm,c.lambda,mu)*G(c.lambda,mu,b.l,mb,c.L,M);
                    v*=8*phase*sign*std::sqrt(4*pi/3)*bohr;
                    if(std::abs(v)>1e-14)append_term(plan,{out,ci,c.L*c.L+M,q+1,v});
                }
            }
        }
    }
}
double interpolate(const Tables& t,const Channel& c,double r) {
    double u=r/t.step;int start=std::clamp(int(std::floor(u))-2,0,t.ndistance-6);
    double v=0;
    for(int j=0;j<6;j++){
        double w=1;
        for(int k=0;k<6;k++)if(k!=j)w*=(u-(start+k))/double(j-k);
        v+=w*c.values[start+j];
    }return v;
}
}
