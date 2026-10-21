extern "C" __global__ void radial_transform(const long long* keys,const long long* ls,const long long* go,const long long* co,
 const double* radius,const double* coeff,const double* gx,const double* gw,
 int order,int nk,double dk,int ntrans,double* out){
 int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=ntrans*nk)return;
 int t=idx/nk,k=idx%nk,a=keys[3*t],l=keys[3*t+1],power=keys[3*t+2];double sum=0;
 for(long long j=go[a];j<go[a+1]-1;j++){
  double lo=radius[j],h=radius[j+1]-lo;const double* c=coeff+co[a]+4*(j-go[a]);
  for(int q=0;q<order;q++){
   double x=(gx[q]+1)*h/2,r=lo+x,f=((c[0]*x+c[1])*x+c[2])*x+c[3],v[8];
   // Baseline calls through max input l+1 for all transforms. This only
   // affects the stable evaluation branch; power=0 requests one extra l.
   spherical_bessel(ls[a]+1,k*dk*r,v);
   sum+=gw[q]*h/2*r*r*f*(power?r:1.)*v[l];
  }
 }out[idx]=sum;
}
extern "C" __global__ void radial_channels(const double* f,const long long* transform_pairs,
 const long long* channel_keys,int nk,double dk,int nr,double dr,int nch,double* out){
 long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=(long long)nch*nr)return;
 int ci=idx/nr,ri=idx%nr,L=channel_keys[5*ci+4];double sum=0;
 int a=transform_pairs[2*ci],b=transform_pairs[2*ci+1];
 for(int k=0;k<nk;k++){
  double v[8];spherical_bessel(7,ri*dr*k*dk,v);
  double w=(k==0||k==nk-1?1:(k%2?4:2))*dk/3*(k*dk)*(k*dk);
  sum+=f[a*nk+k]*f[b*nk+k]*v[L]*w;
 }out[idx]=sum;
}
