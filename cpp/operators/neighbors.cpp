#include "neighbors.hpp"
#include <algorithm>
namespace kubo {
Neighbors neighbors(const double*p,const double*c,const double*inv,const double*cut,I n,I budget){
    Neighbors out;
    for(I i=0;i<n;i++)for(I j=i;j<n;j++){
        double delta[3],df[3]={0,0,0},limit[3];I lo[3],hi[3];double cutoff=cut[i]+cut[j];
        for(int q=0;q<3;q++)delta[q]=p[3*j+q]-p[3*i+q];
        for(int q=0;q<3;q++){
            for(int k=0;k<3;k++)df[q]+=delta[k]*inv[3*k+q];
            limit[q]=cutoff*std::hypot(std::hypot(inv[q],inv[3+q]),inv[6+q]);
            require(std::isfinite(df[q]+limit[q]) && std::abs(df[q])+limit[q]<1e12,"invalid/unbounded periodic image range");
            lo[q]=I(std::ceil(-df[q]-limit[q]));hi[q]=I(std::floor(-df[q]+limit[q]));
        }
        I candidates=1;for(int q=0;q<3;q++)candidates=checked_product(candidates,std::max<I>(0,hi[q]-lo[q]+1));
        require(candidates<budget/64,"periodic neighbor enumeration exceeds budget");
        for(I x=lo[0];x<=hi[0];x++)for(I y=lo[1];y<=hi[1];y++)for(I z=lo[2];z<=hi[2];z++){
            if(i==j && (x<0 || (x==0 && y<0) || (x==0 && y==0 && z<0)))continue;
            std::array<double,3>d;for(int q=0;q<3;q++)d[q]=delta[q]+x*c[q]+y*c[3+q]+z*c[6+q];
            if(std::hypot(std::hypot(d[0],d[1]),d[2])>=cutoff)continue;
            require(I(out.keys.size()/5+1)*64<budget,"neighbor list exceeds budget");
            for(I v:{i,j,x,y,z})out.keys.push_back(v);for(double v:d)out.displacement.push_back(v);
        }
    }return out;
}
}
