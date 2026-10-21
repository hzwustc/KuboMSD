#include "angular.hpp"
#include <map>
#include <mutex>
#include <algorithm>
namespace kubo {
std::pair<std::vector<double>,std::vector<double>> gauss(int n) {
    require(n>=2 && n<=4096,"invalid Gauss order");
    std::vector<double> x(n),w(n);
    for(int i=0;i<(n+1)/2;i++) {
        double z=std::cos(pi*(i+.75)/(n+.5)),p=0,dp=0;
        for(int iter=0;iter<100;iter++) {
            double p0=1,p1=z;
            for(int l=2;l<=n;l++){double pn=((2*l-1)*z*p1-(l-1)*p0)/l;p0=p1;p1=pn;}
            p=p1; dp=n*(z*p1-p0)/(z*z-1); double next=z-p/dp;
            if(std::abs(next-z)<3e-16){z=next;break;} z=next;
        }
        // Re-evaluate derivative at the converged node.
        double p0=1,p1=z;
        for(int l=2;l<=n;l++){double pn=((2*l-1)*z*p1-(l-1)*p0)/l;p0=p1;p1=pn;}
        dp=n*(z*p1-p0)/(z*z-1);
        x[i]=-z;x[n-1-i]=z;w[i]=w[n-1-i]=2/((1-z*z)*dp*dp);
    } return {x,w};
}
std::vector<double> harmonics(int maxl,const std::array<double,3>& d) {
    double r=std::hypot(std::hypot(d[0],d[1]),d[2]);
    double z=r>0?std::clamp(d[2]/r,-1.,1.):1.; double phi=std::atan2(d[1],d[0]);
    std::vector<double> y((maxl+1)*(maxl+1));
    for(int m=0;m<=maxl;m++) {
        double pmm=1;
        for(int k=1;k<=m;k++)pmm*=-(2*k-1)*std::sqrt(std::max(0.,1-z*z));
        double prev=0,p=pmm;
        for(int l=m;l<=maxl;l++) {
            if(l>m){double q=((2*l-1)*z*p-(l+m-1)*prev)/(l-m);prev=p;p=q;}
            double norm=std::sqrt((2*l+1)/(4*pi)*std::exp(std::lgamma(l-m+1.)-std::lgamma(l+m+1.)));
            if(m==0)y[l*l]=norm*p;
            else {y[l*l+2*m-1]=std::sqrt(2.)*norm*p*std::cos(m*phi);y[l*l+2*m]=std::sqrt(2.)*norm*p*std::sin(m*phi);}
        }
    } return y;
}
double gaunt(int la,int ma,int lb,int mb,int lc,int mc) {
    if((la+lb+lc)%2 || lc<std::abs(la-lb) || lc>la+lb)return 0.;
    // Small immutable cubature initialized once before entering pair OpenMP loops.
    static const auto data=[](){
        std::vector<std::pair<double,std::vector<double>>> rows;
        auto g=gauss(20); int np=40;
        for(int i=0;i<20;i++)for(int p=0;p<np;p++){
            double z=g.first[i],phi=2*pi*p/np,s=std::sqrt(1-z*z);
            rows.emplace_back(g.second[i]*2*pi/np,harmonics(7,{s*std::cos(phi),s*std::sin(phi),z}));
        }return rows;
    }();
    double sum=0;
    for(const auto &r:data)sum+=r.first*r.second[la*la+ma]*r.second[lb*lb+mb]*r.second[lc*lc+mc];
    return std::abs(sum)<2e-14?0.:sum;
}
void bessel(int maxl,double x,double* out) {
    if(x==0){out[0]=1;for(int l=1;l<=maxl;l++)out[l]=0;return;}
    if(x<maxl+1.) {
        for(int l=0;l<=maxl;l++) {
            double term=1;for(int j=1;j<=l;j++)term*=x/(2*j+1);
            double sum=term;
            for(int k=1;k<100;k++) {term*=-x*x/(2*k*(2*l+2*k+1));sum+=term;if(std::abs(term)<1e-18*std::max(1.,std::abs(sum)))break;}
            out[l]=sum;
        }
    }else{
        out[0]=std::sin(x)/x;
        if(maxl>0)out[1]=(out[0]-std::cos(x))/x;
        for(int l=1;l<maxl;l++)out[l+1]=(2*l+1)*out[l]/x-out[l-1];
    }
}
}
