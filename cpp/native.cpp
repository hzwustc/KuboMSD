// Independent C++17 kernels for complex nonorthogonal stochastic transport.
// No BLAS/MPI calls: each OpenMP region explicitly uses the configured limit.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace py = pybind11;
using C = std::complex<double>;
using Vec = std::vector<C>;
using Arr = py::array_t<C, py::array::c_style | py::array::forcecast>;
using Real = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Index = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using Clock = std::chrono::steady_clock;

struct Dense {
    std::int64_t n, b;
    Vec x;
    Dense(std::int64_t rows, std::int64_t cols)
        : n(rows), b(cols), x(static_cast<size_t>(rows)*cols) {}
    explicit Dense(const Arr &a) : n(0), b(0) {
        if (a.ndim()!=2)
            throw std::invalid_argument("expected nonempty finite (N,batch) array");
        n=a.shape(0); b=a.shape(1);
        if (n<1 || b<1)
            throw std::invalid_argument("expected nonempty finite (N,batch) array");
        x.assign(a.data(), a.data()+n*b);
        for (C v:x) if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
            throw std::invalid_argument("expected finite (N,batch) array");
    }
};

py::array_t<C> as_array(Dense &&a) {
    py::array_t<C> out({a.n,a.b});
    std::copy(a.x.begin(),a.x.end(),out.mutable_data());
    return out;
}
template<class T> py::array_t<T> array_copy(const std::vector<T> &x) {
    py::array_t<T> a(x.size());
    std::copy(x.begin(),x.end(),a.mutable_data());
    return a;
}

struct CSR {
    std::int64_t n;
    std::vector<std::int64_t> ptr, idx;
    Vec val;
    int threads;
    CSR(Index p, Index i, Arr v, std::int64_t rows, int nt)
        : n(rows), threads(nt) {
        if (nt<1 || n<1 || p.ndim()!=1 || i.ndim()!=1 || v.ndim()!=1
            || p.size()!=n+1 || i.size()!=v.size())
            throw std::invalid_argument("invalid CSR dimensions/threads");
        ptr.assign(p.data(),p.data()+p.size());
        idx.assign(i.data(),i.data()+i.size());
        val.assign(v.data(),v.data()+v.size());
        if (ptr.front()!=0 || ptr.back()!=static_cast<std::int64_t>(val.size()))
            throw std::invalid_argument("invalid CSR row pointers");
        for (std::int64_t row=0;row<n;++row)
            if (ptr[row]>ptr[row+1]) throw std::invalid_argument("nonmonotone CSR row pointers");
        for (size_t k=0;k<idx.size();++k)
            if (idx[k]<0 || idx[k]>=n || !std::isfinite(val[k].real()) || !std::isfinite(val[k].imag()))
                throw std::invalid_argument("invalid CSR index or nonfinite entry");
    }
    Dense mul(const Dense &a) const {
        if (a.n!=n) throw std::invalid_argument("CSR input dimension mismatch");
        Dense out(n,a.b);
        // Tiny arrays stay serial to avoid OpenMP startup dominating their work.
        #pragma omp parallel for num_threads(threads) schedule(static) if(n*a.b>=2048 && threads>1)
        for (std::int64_t r=0;r<n;++r) {
            C *dst=out.x.data()+r*a.b;
            for (auto k=ptr[r];k<ptr[r+1];++k) {
                const C *src=a.x.data()+idx[k]*a.b;
                const C v=val[k];
                for (std::int64_t j=0;j<a.b;++j) dst[j]+=v*src[j];
            }
        }
        return out;
    }
};

// IC(0) uses precisely the sorted lower-triangular sparsity pattern provided.
// The shift affects only this normalized preconditioner, never the physical S.
Vec ic0_factor(const CSR &a,double shift) {
    if (!std::isfinite(shift) || shift<0)throw std::invalid_argument("invalid IC shift");
    Vec values(a.val.size());
    for (std::int64_t i=0;i<a.n;++i) {
        if (a.ptr[i]==a.ptr[i+1] || a.idx[a.ptr[i+1]-1]!=i)
            throw std::invalid_argument("IC pattern requires diagonal at row end");
        for (auto ij=a.ptr[i];ij<a.ptr[i+1]-1;++ij) {
            const auto j=a.idx[ij];
            if (j>=i || (ij>a.ptr[i] && a.idx[ij-1]>=j))
                throw std::invalid_argument("IC pattern must be sorted strict lower triangle");
            C value=a.val[ij];
            auto ik=a.ptr[i],jk=a.ptr[j];
            while (ik<ij && jk<a.ptr[j+1]-1) {
                if (a.idx[ik]<a.idx[jk])++ik;
                else if (a.idx[jk]<a.idx[ik])++jk;
                else {value-=values[ik]*std::conj(values[jk]);++ik;++jk;}
            }
            values[ij]=value/values[a.ptr[j+1]-1];
        }
        C pivot=a.val[a.ptr[i+1]-1]+shift;
        for(auto k=a.ptr[i];k<a.ptr[i+1]-1;++k)pivot-=std::norm(values[k]);
        if (!std::isfinite(pivot.real()) || !std::isfinite(pivot.imag()) ||
            std::abs(pivot.imag())>1e-12 || pivot.real()<=1e-14)
            throw std::runtime_error("IC nonpositive/nonfinite pivot at row "+std::to_string(i));
        values[a.ptr[i+1]-1]=std::sqrt(pivot.real());
    }
    return values;
}

struct SolveReport {
    std::vector<unsigned char> converged, breakdown;
    std::vector<std::int64_t> iterations;
    std::vector<double> residual, relative;
    explicit SolveReport(std::int64_t b)
        : converged(b),breakdown(b),iterations(b),residual(b),relative(b) {}
};
py::dict report_dict(const SolveReport &r) {
    py::dict d;
    py::list cv, br;
    for (auto x:r.converged) cv.append(bool(x));
    for (auto x:r.breakdown) br.append(bool(x));
    d["converged"]=cv; d["breakdown"]=br;
    d["iterations"]=r.iterations; d["true_residual"]=r.residual; d["relative_residual"]=r.relative;
    return d;
}
struct MetricFailure : std::runtime_error {
    SolveReport report;
    MetricFailure(const std::string &msg, SolveReport r)
        : std::runtime_error(msg),report(std::move(r)) {}
};

struct Block {
    std::int64_t a,b;
    Vec L;
    std::vector<std::int64_t> indices;
};
struct Metric {
    std::shared_ptr<CSR> S;
    std::vector<double> diag;
    std::vector<Block> blocks;
    std::shared_ptr<CSR> ic_factor;
    std::vector<std::int64_t> ic_permutation;
    std::vector<double> ic_scaling;
    int threads, observed_threads=1;
    std::int64_t calls=0, iterations=0, refreshes=0, residual_checks=0;
    std::int64_t residual_restarts=0;
    bool adaptive=false;
    int check_interval=32;
    double gap_ratio=.1;
    void set_ic0(std::shared_ptr<CSR> L,Index p,Real scaling) {
        if(L->n!=S->n || p.ndim()!=1 || p.size()!=S->n ||
            scaling.ndim()!=1 || scaling.size()!=S->n)
            throw std::invalid_argument("invalid IC dimensions");
        std::vector<bool> seen(S->n,false);
        for(std::int64_t i=0;i<S->n;++i) {
            auto j=p.data()[i];
            if(j<0 || j>=S->n || seen[j] || !(scaling.data()[i]>0) || !std::isfinite(scaling.data()[i]))
                throw std::invalid_argument("invalid IC permutation/scaling");
            seen[j]=true;
            if(L->ptr[i]==L->ptr[i+1] || L->idx[L->ptr[i+1]-1]!=i ||
                !(L->val[L->ptr[i+1]-1].real()>0))
                throw std::invalid_argument("invalid IC factor diagonal");
        }
        ic_factor=std::move(L);ic_permutation.assign(p.data(),p.data()+p.size());
        ic_scaling.assign(scaling.data(),scaling.data()+scaling.size());
    }
    void set_index_blocks(py::list bs) {
        std::vector<Block> prepared;std::vector<bool> seen(S->n,false);
        for (auto item:bs) {
            auto t=py::cast<py::tuple>(item);Index ids=py::cast<Index>(t[0]);Arr l=py::cast<Arr>(t[1]);
            Block q;q.a=0;q.b=ids.size();
            if (ids.ndim()!=1 || q.b<1 || l.ndim()!=2 || l.shape(0)!=q.b || l.shape(1)!=q.b)
                throw std::invalid_argument("invalid cluster factor");
            q.indices.assign(ids.data(),ids.data()+ids.size());q.L.assign(l.data(),l.data()+l.size());
            for (auto i:q.indices) {
                if(i<0 || i>=S->n || seen[i])throw std::invalid_argument("invalid cluster coverage");
                seen[i]=true;
            }
            prepared.push_back(std::move(q));
        }
        if(std::find(seen.begin(),seen.end(),false)!=seen.end())throw std::invalid_argument("incomplete cluster coverage");
        blocks=std::move(prepared);
    }
    bool direct=false;
    int refine_maxiter=3;
    mutable int direct_solve_workers_observed=0;
    std::int64_t refinement_steps=0;
    void set_direct(bool enabled,int steps) {
        if (steps<0 || (enabled && (blocks.size()!=1 || blocks[0].a!=0 || blocks[0].b!=S->n)))
            throw std::invalid_argument("invalid direct metric factor");
        direct=enabled;refine_maxiter=steps;
    }
    void set_residual_policy(const std::string &policy,int interval,double gap) {
        if ((policy!="fixed" && policy!="adaptive") || interval<1 || !(gap>0 && gap<1))
            throw std::invalid_argument("invalid residual policy");
        adaptive=policy=="adaptive";check_interval=interval;gap_ratio=gap;
    }
    double elapsed=0, max_relative=0;
    Metric(std::shared_ptr<CSR> s, Real d, py::list bs, int nt)
        : S(std::move(s)),threads(nt) {
        if (d.ndim()!=1 || d.size()!=S->n || nt<1) throw std::invalid_argument("invalid metric diagonal/threads");
        diag.assign(d.data(),d.data()+d.size());
        for (double x:diag) if (!(x>0) || !std::isfinite(x)) throw std::invalid_argument("invalid metric diagonal");
        std::int64_t next=0;
        for (auto item:bs) {
            auto t=py::cast<py::tuple>(item);
            Block q;
            q.a=py::cast<std::int64_t>(t[0]); q.b=py::cast<std::int64_t>(t[1]);
            Arr l=py::cast<Arr>(t[2]);
            if (q.a!=next || q.b<=q.a || q.b>S->n || l.ndim()!=2
                || l.shape(0)!=q.b-q.a || l.shape(1)!=q.b-q.a)
                throw std::invalid_argument("invalid block-Jacobi factor partition");
            q.L.assign(l.data(),l.data()+l.size());
            blocks.push_back(std::move(q)); next=blocks.back().b;
        }
        if (!blocks.empty() && next!=S->n) throw std::invalid_argument("incomplete block-Jacobi partition");
        // Report runtime-limited team size, not the hardware CPU count.
        #pragma omp parallel num_threads(threads)
        {
            #pragma omp single
            observed_threads=omp_get_num_threads();
        }
    }
    Dense precondition(const Dense &r) const {
        Dense z(r.n,r.b);
        if (ic_factor) {
            Dense work(r.n,r.b);const auto &L=*ic_factor;
            #pragma omp parallel for num_threads(threads) schedule(static) if(r.n*r.b>=2048 && threads>1)
            for(std::int64_t j=0;j<r.b;++j) {
                for(std::int64_t i=0;i<r.n;++i) {
                    auto original=ic_permutation[i];
                    C value=ic_scaling[original]*r.x[original*r.b+j];
                    for(auto k=L.ptr[i];k<L.ptr[i+1]-1;++k)value-=L.val[k]*work.x[L.idx[k]*r.b+j];
                    work.x[i*r.b+j]=value/L.val[L.ptr[i+1]-1];
                }
                for(std::int64_t i=r.n;i-->0;) {
                    C value=work.x[i*r.b+j]/std::conj(L.val[L.ptr[i+1]-1]);
                    work.x[i*r.b+j]=value;
                    for(auto k=L.ptr[i];k<L.ptr[i+1]-1;++k)work.x[L.idx[k]*r.b+j]-=std::conj(L.val[k])*value;
                    auto original=ic_permutation[i];
                    z.x[original*r.b+j]=ic_scaling[original]*value;
                }
            }
            return z;
        }
        if (blocks.empty()) {
            #pragma omp parallel for num_threads(threads) schedule(static) if(r.n*r.b>=4096 && threads>1)
            for (std::int64_t i=0;i<r.n;++i)
                for (std::int64_t j=0;j<r.b;++j) z.x[i*r.b+j]=r.x[i*r.b+j]/diag[i];
            return z;
        }
        auto solve_column=[&](const Block &bl,std::int64_t j) {
            const auto m=bl.b-bl.a;
            auto index=[&](std::int64_t i){return bl.indices.empty() ? bl.a+i : bl.indices[i];};
            for (std::int64_t i=0;i<m;++i) {
                C v=r.x[index(i)*r.b+j];
                for (std::int64_t k=0;k<i;++k) v-=bl.L[i*m+k]*z.x[index(k)*r.b+j];
                z.x[index(i)*r.b+j]=v/bl.L[i*m+i];
            }
            for (std::int64_t i=m;i-->0;) {
                C v=z.x[index(i)*r.b+j];
                for (std::int64_t k=i+1;k<m;++k) v-=std::conj(bl.L[k*m+i])*z.x[index(k)*r.b+j];
                z.x[index(i)*r.b+j]=v/std::conj(bl.L[i*m+i]);
            }
        };
        if (direct) {
            // A full Cholesky factor is one block. Distribute independent RHS
            // columns, retaining exactly the serial summation order per column.
            std::vector<int> used(threads,0);
            #pragma omp parallel for num_threads(threads) schedule(static) if(r.n*r.b>=2048 && threads>1)
            for (std::int64_t j=0;j<r.b;++j) {
                used[omp_get_thread_num()]=1;
                solve_column(blocks[0],j);
            }
            direct_solve_workers_observed=std::max(direct_solve_workers_observed,
                static_cast<int>(std::count(used.begin(),used.end(),1)));
            return z;
        }
        #pragma omp parallel for num_threads(threads) schedule(static) if(r.n*r.b>=2048 && threads>1)
        for (std::int64_t bi=0;bi<static_cast<std::int64_t>(blocks.size());++bi) {
            for (std::int64_t j=0;j<r.b;++j) solve_column(blocks[bi],j);
        }
        return z;
    }
    // Column reductions are serial in row order, preserving repeatability across
    // OpenMP thread counts. SpMM and independent vector rows carry parallelism.
    std::vector<C> dot(const Dense &a,const Dense &b) const {
        std::vector<C> v(a.b);
        for (std::int64_t i=0;i<a.n;++i)
            for (std::int64_t j=0;j<a.b;++j) v[j]+=std::conj(a.x[i*a.b+j])*b.x[i*a.b+j];
        return v;
    }
    std::vector<double> norms(const Dense &a) const {
        std::vector<double> v(a.b);
        for (std::int64_t i=0;i<a.n;++i)
            for (std::int64_t j=0;j<a.b;++j) v[j]+=std::norm(a.x[i*a.b+j]);
        for (auto &x:v) x=std::sqrt(x);
        return v;
    }
    std::pair<Dense,SolveReport> solve(const Dense &B,double rtol,double atol,int maxiter) {
        if (B.n!=S->n || B.b<1 || !std::isfinite(rtol) || !std::isfinite(atol)
            || rtol<0 || atol<0 || maxiter<1)
            throw std::invalid_argument("invalid metric RHS shape/tolerances");
        const auto start=Clock::now();
        const auto n=B.n,b=B.b;
        std::vector<double> safe(b,0.),threshold(b),rz(b),alpha(b),beta(b);
        for (std::int64_t i=0;i<n;++i) for (std::int64_t j=0;j<b;++j) {
            const C value=B.x[i*b+j];
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                throw std::invalid_argument("metric RHS must be finite");
            // abs(complex) may overflow even when both components are finite.
            safe[j]=std::max(safe[j],std::max(std::abs(value.real()),std::abs(value.imag())));
        }
        for (auto &s:safe) if (s==0.) s=1.;
        Dense q(n,b),y(n,b);
        for (std::int64_t i=0;i<n;++i) for (std::int64_t j=0;j<b;++j) q.x[i*b+j]=B.x[i*b+j]/safe[j];
        auto bn=norms(q);
        std::vector<unsigned char> active(b);
        for (std::int64_t j=0;j<b;++j) {
            threshold[j]=std::max(rtol*bn[j],atol/safe[j]);
            active[j]=bn[j]>threshold[j];
        }
        Dense r=q,z=direct ? Dense(n,b) : precondition(r),p=z;
        auto rz0=dot(r,z);
        for (std::int64_t j=0;j<b;++j) rz[j]=rz0[j].real();
        for (std::int64_t i=0;i<n;++i) for (std::int64_t j=0;j<b;++j)
            if (!active[j]) p.x[i*b+j]=0.;
        SolveReport report(b);
        bool breakdown=false;
        if (direct) {
            y=precondition(q);
            for (int step=0;step<refine_maxiter;++step) {
                auto truth=S->mul(y);++refreshes;++residual_checks;
                for (size_t i=0;i<truth.x.size();++i) truth.x[i]=q.x[i]-truth.x[i];
                auto tn=norms(truth);bool needed=false;
                for (std::int64_t j=0;j<b;++j) {
                    needed |= tn[j]>threshold[j];
                    if (tn[j]<=threshold[j]) for (std::int64_t i=0;i<n;++i) truth.x[i*b+j]=0;
                }
                if (!needed) break;
                auto correction=precondition(truth);
                for (size_t i=0;i<y.x.size();++i) y.x[i]+=correction.x[i];
                ++refinement_steps;
            }
        }
        for (int k=0;k<(direct ? 0 : maxiter);++k) {
            if (std::none_of(active.begin(),active.end(),[](auto x){return x!=0;})) break;
            Dense sp=S->mul(p);
            auto curvature=dot(p,sp);
            for (std::int64_t j=0;j<b;++j) {
                if (active[j] && (!(curvature[j].real()>0) || !std::isfinite(curvature[j].real())
                    || !std::isfinite(curvature[j].imag())
                    || std::abs(curvature[j].imag())>1e-10*std::max(std::abs(curvature[j].real()),std::numeric_limits<double>::min()))) {
                    report.breakdown[j]=1; breakdown=true;
                }
                alpha[j]=active[j] ? rz[j]/curvature[j].real() : 0.;
            }
            if (breakdown) break;
            #pragma omp parallel for num_threads(threads) schedule(static) if(n*b>=4096 && threads>1)
            for (std::int64_t i=0;i<n;++i) for (std::int64_t j=0;j<b;++j) {
                y.x[i*b+j]+=p.x[i*b+j]*alpha[j];
                r.x[i*b+j]-=sp.x[i*b+j]*alpha[j];
            }
            for (std::int64_t j=0;j<b;++j) report.iterations[j]+=active[j];
            auto rn=norms(r);
            const bool refresh=(k+1)%check_interval==0;
            std::vector<unsigned char> restart(b,0);
            bool candidate=false;
            for (std::int64_t j=0;j<b;++j) candidate |= active[j] && rn[j]<=threshold[j];
            if (refresh || (adaptive && candidate)) {
                auto truth=S->mul(y); ++refreshes;++residual_checks;
                for (size_t i=0;i<truth.x.size();++i) truth.x[i]=q.x[i]-truth.x[i];
                auto tn=norms(truth);
                for (std::int64_t j=0;j<b;++j) if (active[j]) {
                    if (adaptive) {
                        double gap=0;
                        for (std::int64_t i=0;i<n;++i) gap+=std::norm(truth.x[i*b+j]-r.x[i*b+j]);
                        restart[j]=tn[j]>threshold[j] &&
                            (std::sqrt(gap)>gap_ratio*std::max(tn[j],threshold[j]) || rn[j]<=threshold[j]);
                        active[j]=tn[j]>threshold[j];
                    } else {restart[j]=1;active[j]=tn[j]>threshold[j];}
                    if (restart[j]) {
                        ++residual_restarts;
                        for (std::int64_t i=0;i<n;++i) r.x[i*b+j]=truth.x[i*b+j];
                    }
                }
            } else if (!adaptive) {
                for (std::int64_t j=0;j<b;++j) active[j]=active[j] && rn[j]>threshold[j];
            }
            z=precondition(r);
            auto newrz=dot(r,z);
            for (std::int64_t j=0;j<b;++j) {
                beta[j]=active[j] ? newrz[j].real()/(rz[j]!=0 ? rz[j] : 1.) : 0.;
                rz[j]=newrz[j].real();
            }
            #pragma omp parallel for num_threads(threads) schedule(static) if(n*b>=4096 && threads>1)
            for (std::int64_t i=0;i<n;++i) for (std::int64_t j=0;j<b;++j)
                p.x[i*b+j]=active[j] ? z.x[i*b+j]+(restart[j] ? C(0) : p.x[i*b+j]*beta[j]) : C(0);
        }
        Dense sy=S->mul(y);++residual_checks;
        for (size_t i=0;i<r.x.size();++i) r.x[i]=q.x[i]-sy.x[i];
        const auto rn=norms(r);
        bool ok=!breakdown;
        for (std::int64_t j=0;j<b;++j) {
            report.converged[j]=rn[j]<=threshold[j] && !report.breakdown[j];
            report.relative[j]=bn[j]>0 ? rn[j]/bn[j] : 0.;
            report.residual[j]=rn[j]*safe[j];
            max_relative=std::max(max_relative,report.relative[j]);
            ok=ok && report.converged[j];
        }
        std::vector<unsigned char> finite_solution(b,1);
        for (std::int64_t i=0;i<n;++i) for (std::int64_t j=0;j<b;++j) {
            const C value=y.x[i*b+j];
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                finite_solution[j]=0;
        }
        // Check each column both before and after restoring physical RHS scale.
        // The scaled equation can converge even when its solution exceeds
        // complex128's component range after rescaling.
        for (std::int64_t i=0;i<n;++i) for (std::int64_t j=0;j<b;++j) if (finite_solution[j]) {
            y.x[i*b+j]*=safe[j];
            const C value=y.x[i*b+j];
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                finite_solution[j]=0;
        }
        bool representable=true;
        for (std::int64_t j=0;j<b;++j) {
            report.converged[j]=report.converged[j] && finite_solution[j];
            representable=representable && finite_solution[j];
        }
        ok=ok && representable;
        ++calls; iterations+=*std::max_element(report.iterations.begin(),report.iterations.end());
        elapsed+=std::chrono::duration<double>(Clock::now()-start).count();
        if (!ok) throw MetricFailure(!representable ?
                                     "S solve solution is nonfinite or not representable in complex128 after rescaling" :
                                     breakdown ? "PCG non-positive/complex curvature breakdown" :
                                     "S solve failed true-residual acceptance",std::move(report));
        return {std::move(y),std::move(report)};
    }
    Dense sample(const Dense &xi,const std::vector<double> &coeff,double center,double half) const {
        if (xi.n!=S->n || coeff.empty()) throw std::invalid_argument("invalid sampling dimensions");
        std::vector<double> z(diag.size());
        for (size_t i=0;i<z.size();++i) z[i]=1/std::sqrt(diag[i]);
        auto action=[&](const Dense &a) {
            Dense scaled=a;
            for (std::int64_t i=0;i<a.n;++i) for (std::int64_t j=0;j<a.b;++j) scaled.x[i*a.b+j]*=z[i];
            Dense out=S->mul(scaled);
            for (std::int64_t i=0;i<a.n;++i) for (std::int64_t j=0;j<a.b;++j)
                out.x[i*a.b+j]=(z[i]*out.x[i*a.b+j]-center*a.x[i*a.b+j])/half;
            return out;
        };
        Dense b1(xi.n,xi.b),b2(xi.n,xi.b),out=xi;
        if (coeff.size()>1) {
            if (!(half>0)) throw std::invalid_argument("sampling half-width must be positive");
            for (size_t k=coeff.size();--k>0;) {
                Dense b0=action(b1);
                for (size_t i=0;i<b0.x.size();++i) b0.x[i]=2.*b0.x[i]-b2.x[i]+coeff[k]*xi.x[i];
                b2=std::move(b1); b1=std::move(b0);
            }
            out=action(b1);
            for (size_t i=0;i<out.x.size();++i) out.x[i]-=b2.x[i];
        } else std::fill(out.x.begin(),out.x.end(),0.);
        for (std::int64_t i=0;i<xi.n;++i) for (std::int64_t j=0;j<xi.b;++j)
            out.x[i*xi.b+j]=z[i]*(out.x[i*xi.b+j]+coeff[0]*xi.x[i*xi.b+j]);
        return out;
    }
};

// Cartesian operator kind: 0 explicit covariant velocity, 1 position,
// 2 implicit periodic bond/xbar/delta. Coefficients remain physical A/fs.
struct Velocity {
    int kind;
    std::vector<std::shared_ptr<CSR>> matrices;
};
struct Engine {
    std::shared_ptr<CSR> H;
    std::shared_ptr<Metric> metric;
    std::vector<Velocity> cartesian;
    double rtol,atol,hbar;
    int maxiter;
    std::int64_t h_calls=0,v_calls=0;
    Engine(std::shared_ptr<CSR> h,std::shared_ptr<Metric> m,py::list definitions,
           double rt,double at,int mi,double hb)
        : H(std::move(h)),metric(std::move(m)),rtol(rt),atol(at),hbar(hb),maxiter(mi) {
        for (auto def:definitions) {
            auto t=py::cast<py::tuple>(def);
            cartesian.push_back({py::cast<int>(t[0]),py::cast<std::vector<std::shared_ptr<CSR>>>(t[1])});
        }
        if (cartesian.size()!=3 || H->n!=metric->S->n) throw std::invalid_argument("invalid engine operators");
    }
    Dense solve(const Dense &B) { return metric->solve(B,rtol,atol,maxiter).first; }
    Dense h_action(const Dense &B) { ++h_calls; return solve(H->mul(B)); }
    Dense v_cart(int axis,const Dense &B) {
        const auto &v=cartesian.at(axis); const auto &m=v.matrices;
        if (v.kind==0 && m.size()==1) return m[0]->mul(B);
        if (v.kind==1 && m.size()==1) {
            auto out=H->mul(solve(m[0]->mul(B)));
            auto rhs=m[0]->mul(solve(H->mul(B)));
            for (size_t i=0;i<out.x.size();++i) out.x[i]=C(0,1)*(out.x[i]-rhs.x[i])/hbar;
            return out;
        }
        if (v.kind==2 && m.size()==3) {
            auto out=m[0]->mul(B);
            auto left=m[1]->mul(solve(H->mul(B)));
            auto right=H->mul(solve(m[2]->mul(B)));
            for (size_t i=0;i<out.x.size();++i) out.x[i]=C(0,1)*(out.x[i]-left.x[i]+right.x[i])/hbar;
            return out;
        }
        throw std::invalid_argument("missing Cartesian velocity");
    }
    Dense v_cov(const std::vector<double> &direction,const Dense &B) {
        if (direction.size()!=3) throw std::invalid_argument("direction must have three components");
        ++v_calls;
        Dense out(B.n,B.b);
        for (int a=0;a<3;++a) if (direction[a]!=0.) {
            auto part=v_cart(a,B);
            for (size_t i=0;i<out.x.size();++i) out.x[i]+=direction[a]*part.x[i];
        }
        return out;
    }
    Dense scaled(const Dense &B,double center,double half) {
        if (!(half>0)) throw std::invalid_argument("spectral half-width must be positive");
        auto out=h_action(B);
        for (size_t i=0;i<out.x.size();++i) out.x[i]=(out.x[i]-center*B.x[i])/half;
        return out;
    }
    Dense augmented(const Dense &B,double center,double half,const std::vector<std::vector<double>> &directions) {
        Dense out=scaled(B,center,half);
        if (directions.empty()) return out;
        if (B.b%static_cast<std::int64_t>(directions.size()+1)) throw std::invalid_argument("invalid augmented block columns");
        const auto nb=B.b/static_cast<std::int64_t>(directions.size()+1);
        Dense initial(B.n,nb);
        for (std::int64_t i=0;i<B.n;++i)
            std::copy_n(B.x.data()+i*B.b,nb,initial.x.data()+i*nb);
        for (size_t a=0;a<directions.size();++a) {
            auto v=solve(v_cov(directions[a],initial));
            for (std::int64_t i=0;i<B.n;++i) for (std::int64_t j=0;j<nb;++j)
                out.x[i*B.b+(a+1)*nb+j]+=C(0,-hbar/half)*v.x[i*nb+j];
        }
        return out;
    }
    Dense series(const Dense &B,const Vec &coeff,double center,double half,
                 const std::vector<std::vector<double>> &directions) {
        if (coeff.empty()) throw std::invalid_argument("empty Chebyshev coefficients");
        Dense out=B;
        for (auto &v:out.x) v*=coeff[0];
        if (coeff.size()==1) return out;
        Dense prev=B,curr=augmented(B,center,half,directions);
        for (size_t i=0;i<out.x.size();++i) out.x[i]+=coeff[1]*curr.x[i];
        for (size_t k=2;k<coeff.size();++k) {
            Dense next=augmented(curr,center,half,directions);
            for (size_t i=0;i<out.x.size();++i) {
                next.x[i]=2.*next.x[i]-prev.x[i]; out.x[i]+=coeff[k]*next.x[i];
            }
            prev=std::move(curr);curr=std::move(next);
        }
        for (C value:out.x) if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            throw std::runtime_error("nonfinite Chebyshev propagation; check bounds and solves");
        return out;
    }
    Vec chain(const Dense &B,std::int64_t length,double center,double half) {
        if (length<1) throw std::invalid_argument("Chebyshev chain must be nonempty");
        const size_t stride=B.x.size();
        Vec out(stride*length);
        std::copy(B.x.begin(),B.x.end(),out.begin());
        if (length==1) return out;
        Dense prev=B,curr=scaled(B,center,half);
        std::copy(curr.x.begin(),curr.x.end(),out.begin()+stride);
        for (std::int64_t k=2;k<length;++k) {
            Dense next=scaled(curr,center,half);
            for (size_t i=0;i<stride;++i) next.x[i]=2.*next.x[i]-prev.x[i];
            std::copy(next.x.begin(),next.x.end(),out.begin()+k*stride);
            prev=std::move(curr);curr=std::move(next);
        }
        return out;
    }
    std::vector<double> moments(const Dense &B,std::int64_t length,double center,double half) {
        if (length<1) throw std::invalid_argument("moment count must be positive");
        std::vector<double> out(length*B.b);
        Dense sb=metric->S->mul(B);
        auto store=[&](std::int64_t k,const Dense &a) {
            auto values=metric->dot(sb,a);
            for (std::int64_t j=0;j<B.b;++j) out[k*B.b+j]=values[j].real();
        };
        store(0,B);
        if (length==1) return out;
        Dense prev=B,curr=scaled(B,center,half);
        store(1,curr);
        for (std::int64_t k=2;k<length;++k) {
            Dense next=scaled(curr,center,half);
            for (size_t i=0;i<next.x.size();++i) next.x[i]=2.*next.x[i]-prev.x[i];
            store(k,next);
            prev=std::move(curr);curr=std::move(next);
        }
        return out;
    }
};

static PyObject *metric_exception_type=nullptr;
PYBIND11_MODULE(_native,m) {
    m.def("ic0_factor",[](const CSR &a,double shift) {
        Vec result;
        {py::gil_scoped_release release;result=ic0_factor(a,shift);}
        return array_copy(result);
    });
    m.doc()="Independent C++17/OpenMP complex CSR, PCG and fused stochastic recurrences";
    auto exception=py::exception<MetricFailure>(m,"MetricFailure",PyExc_RuntimeError);
    metric_exception_type=exception.ptr();
    py::register_local_exception_translator([](std::exception_ptr p) {
        try { if (p) std::rethrow_exception(p); }
        catch (const MetricFailure &e) {
            auto type=py::reinterpret_borrow<py::object>(metric_exception_type);
            py::object value=type(e.what());
            value.attr("diagnostics")=report_dict(e.report);
            PyErr_SetObject(metric_exception_type,value.ptr());
        }
    });
    m.def("build_info",[]() {
        py::dict d;
        d["language"]="C++17"; d["compiler"]=__VERSION__;
        d["openmp"]=true; d["openmp_version"]=_OPENMP;
        d["precision"]="complex128/float64"; d["gil_released"]=true;
        d["architecture_flags"]="portable; no -march=native";
        d["implementation"]="independent CSR/PCG/Chebyshev; no RSATB or BLAS calls";
        return d;
    });
    py::class_<CSR,std::shared_ptr<CSR>>(m,"CSR")
        .def(py::init<Index,Index,Arr,std::int64_t,int>())
        .def("matmul",[](const CSR &s,Arr input) {
            Dense a(input),out(a.n,a.b);
            { py::gil_scoped_release release;out=s.mul(a); }
            return as_array(std::move(out));
        });
    py::class_<Metric,std::shared_ptr<Metric>>(m,"Metric")
        .def(py::init<std::shared_ptr<CSR>,Real,py::list,int>())
        .def("solve",[](Metric &m,Arr input,double rt,double at,int mi) {
            Dense a(input),out(a.n,a.b);SolveReport report(a.b);
            { py::gil_scoped_release release;
              auto result=m.solve(a,rt,at,mi);out=std::move(result.first);report=std::move(result.second); }
            return py::make_tuple(as_array(std::move(out)),report_dict(report));
        })
        .def("sample",[](const Metric &m,Arr input,Real coefficients,double center,double half) {
            Dense a(input),out(a.n,a.b);
            if (coefficients.ndim()!=1) throw std::invalid_argument("sampling coefficients must be 1D");
            std::vector<double> coeff(coefficients.data(),coefficients.data()+coefficients.size());
            { py::gil_scoped_release release;out=m.sample(a,coeff,center,half); }
            return as_array(std::move(out));
        })
        .def("set_residual_policy",&Metric::set_residual_policy)
        .def("set_direct",&Metric::set_direct)
        .def("set_index_blocks",&Metric::set_index_blocks)
        .def("set_ic0",&Metric::set_ic0)
        .def("diagnostics",[](const Metric &m) {
            py::dict d;
            d["calls"]=m.calls;d["max_iterations_sum"]=m.iterations;d["seconds"]=m.elapsed;
            d["max_relative_residual"]=m.max_relative;d["true_residual_refreshes"]=m.refreshes;
            d["true_residual_checks"]=m.residual_checks;d["residual_restarts"]=m.residual_restarts;
            d["refinement_steps"]=m.refinement_steps;
            d["direct_solve_workers_observed"]=m.direct_solve_workers_observed;
            d["omp_threads_requested"]=m.threads;d["omp_threads_observed"]=m.observed_threads;
            return d;
        });
    py::class_<Engine,std::shared_ptr<Engine>>(m,"Engine")
        .def(py::init<std::shared_ptr<CSR>,std::shared_ptr<Metric>,py::list,double,double,int,double>())
        .def("h_action",[](Engine &e,Arr input) {
            Dense a(input),out(a.n,a.b);
            { py::gil_scoped_release release;out=e.h_action(a); }
            return as_array(std::move(out));
        })
        .def("v_cov",[](Engine &e,Arr input,const std::vector<double> &direction) {
            Dense a(input),out(a.n,a.b);
            { py::gil_scoped_release release;out=e.v_cov(direction,a); }
            return as_array(std::move(out));
        })
        .def("v_cov_with_controls",[](Engine &e,Arr input,const std::vector<double> &direction,
                                      double rtol,double atol,int maxiter) {
            Dense a(input),out(a.n,a.b);
            // Reuse immutable matrices and the prepared metric. The transport
            // engine retains its original controls even if validation throws.
            Engine probe=e;probe.rtol=rtol;probe.atol=atol;probe.maxiter=maxiter;
            ++e.v_calls;
            { py::gil_scoped_release release;out=probe.v_cov(direction,a); }
            return as_array(std::move(out));
        })
        .def("series",[](Engine &e,Arr input,Arr coefficients,double center,double half,
                         const std::vector<std::vector<double>> &directions) {
            Dense a(input),out(a.n,a.b);
            if (coefficients.ndim()!=1) throw std::invalid_argument("Chebyshev coefficients must be 1D");
            Vec coeff(coefficients.data(),coefficients.data()+coefficients.size());
            { py::gil_scoped_release release;out=e.series(a,coeff,center,half,directions); }
            return as_array(std::move(out));
        })
        .def("chain",[](Engine &e,Arr input,std::int64_t length,double center,double half) {
            Dense a(input);Vec out;
            { py::gil_scoped_release release;out=e.chain(a,length,center,half); }
            py::array_t<C> result({length,a.n,a.b});
            std::copy(out.begin(),out.end(),result.mutable_data());return result;
        })
        .def("moments",[](Engine &e,Arr input,std::int64_t length,double center,double half) {
            Dense a(input);std::vector<double> out;
            { py::gil_scoped_release release;out=e.moments(a,length,center,half); }
            py::array_t<double> result({length,a.b});
            std::copy(out.begin(),out.end(),result.mutable_data());return result;
        })
        .def_property_readonly("h_calls",[](const Engine &e){return e.h_calls;})
        .def_property_readonly("v_calls",[](const Engine &e){return e.v_calls;});
}
