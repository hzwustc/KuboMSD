// Double precision Fourier-Bessel / real harmonics, ABI frozen from CPU baseline.
// NVRTC provides device math builtins; no host libc header dependency.
#define PI 3.141592653589793238462643383279502884
__device__ void spherical_bessel(int maxl,double x,double* out){
 if(x==0){out[0]=1;for(int l=1;l<=maxl;l++)out[l]=0;return;}
 if(x<maxl+1.){
  for(int l=0;l<=maxl;l++){
   double term=1;for(int j=1;j<=l;j++)term*=x/(2*j+1);double sum=term;
   for(int k=1;k<100;k++){term*=-x*x/(2*k*(2*l+2*k+1));sum+=term;if(fabs(term)<1e-18*fmax(1.,fabs(sum)))break;}out[l]=sum;
  }
 }else{
  out[0]=sin(x)/x;if(maxl>0)out[1]=(out[0]-cos(x))/x;
  for(int l=1;l<maxl;l++)out[l+1]=(2*l+1)*out[l]/x-out[l-1];
 }
}
__device__ void real_angular(double dx,double dy,double dz,double* y){
 double r=hypot(hypot(dx,dy),dz),z=r>0?fmin(1.,fmax(-1.,dz/r)):1.,phi=atan2(dy,dx);
 for(int m=0;m<=7;m++){
  double pmm=1;for(int k=1;k<=m;k++)pmm*=-(2*k-1)*sqrt(fmax(0.,1-z*z));double prev=0,p=pmm;
  for(int l=m;l<=7;l++){
   if(l>m){double q=((2*l-1)*z*p-(l+m-1)*prev)/(l-m);prev=p;p=q;}
   double norm=sqrt((2*l+1)/(4*PI)*exp(lgamma(l-m+1.)-lgamma(l+m+1.)));
   if(m==0)y[l*l]=norm*p;else{y[l*l+2*m-1]=sqrt(2.)*norm*p*cos(m*phi);y[l*l+2*m]=sqrt(2.)*norm*p*sin(m*phi);}
  }
 }
}
