// No floating-point atomics: stable keys followed by serial per-key summation.
extern "C" __global__ void fill_pair_coo(const long long* keys,const long long* offsets,const long long* atom_offsets,
 const long long* ids,const double* displacement,const double* S,const double* D,int np,long long total,
 long long* rows,long long* cols,long long* seq,double2* values){
 long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=2*total)return;
 int side=idx/total;long long at=idx%total;int lo=0,hi=np;
 while(lo+1<hi){int mid=(lo+hi)/2;if(offsets[mid]<=at)lo=mid;else hi=mid;}int p=lo;
 const long long* key=keys+5*p;int i=key[0],j=key[1],nj=atom_offsets[j+1]-atom_offsets[j];
 long long local=at-offsets[p],r=atom_offsets[i]+local/nj,c=atom_offsets[j]+local%nj;
 bool onsite=(i==j&&key[2]==0&&key[3]==0&&key[4]==0);bool keep=!(onsite&&side);
 rows[idx]=side?c:r;cols[idx]=side?r:c;seq[idx]=2*ids[p]+side;
 double s=keep?S[at]:0;values[idx]=make_double2(s,0);
 for(int a=0;a<3;a++){
  double d=displacement[3*p+a],v=keep?D[a*total+at]:0;
  values[(1+a)*2*total+idx]=make_double2(side?v-d*s:v,0);
  values[(4+a)*2*total+idx]=make_double2(side?v:v-d*s,0);
 }
}
extern "C" __global__ void fill_h_coo(const long long* keys,const long long* offsets,const long long* atom_offsets,
 const double* displacement,const double2* data,int np,long long total,long long* rows,long long* cols,long long* seq,double2* out){
 long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=total)return;
 int lo=0,hi=np;while(lo+1<hi){int mid=(lo+hi)/2;if(offsets[mid]<=idx)lo=mid;else hi=mid;}int p=lo;
 int i=keys[5*p],j=keys[5*p+1],nj=atom_offsets[j+1]-atom_offsets[j];long long k=idx-offsets[p];
 rows[idx]=atom_offsets[i]+k/nj;cols[idx]=atom_offsets[j]+k%nj;seq[idx]=p;out[idx]=data[idx];
 for(int a=0;a<3;a++){double d=displacement[3*p+a];out[(a+1)*total+idx]=make_double2(data[idx].x*d,data[idx].y*d);}
}
extern "C" __global__ void reduce_segments(const double2* vals,const long long* order,const long long* starts,
 int segments,long long count,double2* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=segments)return;
 long long end=i+1<segments?starts[i+1]:count;double re=0,im=0;
 for(long long k=starts[i];k<end;k++){double2 x=vals[order[k]];re+=x.x;im+=x.y;}out[i]=make_double2(re,im);
}
