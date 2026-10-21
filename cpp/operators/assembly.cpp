#include "assembly.hpp"
#include <algorithm>
#include <numeric>
namespace kubo {
CSR csr(const I*row,const I*col,const C*data,I count,I n,I budget){
    require(n>0 && n<INT64_MAX/8,"invalid CSR dimension");
    require(checked_product(count,48)+checked_product(n+1,8)<budget,"CSR assembly exceeds memory budget");
    std::vector<I>order(count);std::iota(order.begin(),order.end(),0);
    for(I j=0;j<count;j++)require(row[j]>=0 && row[j]<n && col[j]>=0 && col[j]<n && std::isfinite(data[j].real()) && std::isfinite(data[j].imag()),"invalid COO index/value");
    std::stable_sort(order.begin(),order.end(),[&](I a,I b){return row[a]<row[b] || (row[a]==row[b] && col[a]<col[b]);});
    CSR out;out.indptr.resize(n+1,0);
    for(I start=0;start<count;){
        I r=row[order[start]],c=col[order[start]],end=start+1;C value=data[order[start]];
        while(end<count && row[order[end]]==r && col[order[end]]==c){value+=data[order[end]];end++;}
        if(value!=C(0)){out.indices.push_back(c);out.data.push_back(value);out.indptr[r+1]++;}start=end;
    }
    for(I r=0;r<n;r++)out.indptr[r+1]+=out.indptr[r];return out;
}
}
