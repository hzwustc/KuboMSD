extern "C" __global__ void pair_angular(const double* displacement,int np,double* angular,double* distances){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=np)return;double d[3];for(int a=0;a<3;a++)d[a]=displacement[3*p+a]/0.529177210903;
 real_angular(d[0],d[1],d[2],angular+64*p);distances[p]=hypot(hypot(d[0],d[1]),d[2]);
}
extern "C" __global__ void pair_interpolate(const double* table,const double* distances,int np,int nc,int nr,double step,double* out){
 long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=(long long)np*nc)return;
 int p=idx/nc,c=idx%nc;double u=distances[p]/step;
 if(u>nr-1){out[idx]=0;return;}
 int start=max(0,min((int)floor(u)-2,nr-6));double value=0;
 for(int j=0;j<6;j++){double w=1;for(int k=0;k<6;k++)if(k!=j)w*=(u-(start+k))/(j-k);value+=w*table[c*nr+start+j];}out[idx]=value;
}
extern "C" __global__ void pair_contract(const long long* species,const long long* offsets,int np,int ns,
 const long long* group_offsets,const double* cutoffs,const long long* term_offsets,const long long* channels,
 const long long* harmonics,const double* coefficient,const double* radial,const double* angular,const double* distances,
 int nch,long long total,double* out){
 long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=4*total)return;
 int component=idx/total;long long at=idx%total;int lo=0,hi=np;
 while(lo+1<hi){int mid=(lo+hi)/2;if(offsets[mid]<=at)lo=mid;else hi=mid;}int p=lo;
 int group=species[2*p]*ns+species[2*p+1];if(distances[p]>=cutoffs[group]){out[idx]=0;return;}
 long long size=offsets[p+1]-offsets[p],oi=group_offsets[group]+component*size+at-offsets[p];double sum=0;
 for(long long j=term_offsets[oi];j<term_offsets[oi+1];j++)sum+=coefficient[j]*angular[p*64+harmonics[j]]*radial[p*nch+channels[j]];
 out[idx]=sum;
}
