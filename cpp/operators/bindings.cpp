#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include <set>
#include "types.hpp"
#include "radial_tables.hpp"
#include "pair_integrals.hpp"
#include "neighbors.hpp"
#include "assembly.hpp"
namespace py=pybind11;
using namespace kubo;
template<class T>using Array=py::array_t<T,py::array::c_style|py::array::forcecast>;
template<class T>py::array_t<T> copy_array(const std::vector<T>& v){py::array_t<T>a(v.size());std::copy(v.begin(),v.end(),a.mutable_data());return a;}
Settings read_settings(py::dict d){
    const std::vector<std::string>allowed{"kmax","dk","distance_step","radial_order","memory_limit_bytes"};
    for(auto entry:d){auto key=py::cast<std::string>(entry.first);require(std::find(allowed.begin(),allowed.end(),key)!=allowed.end(),"unknown table setting: "+key);}
    Settings s;
    if(d.contains("kmax"))s.kmax=py::cast<double>(d["kmax"]);
    if(d.contains("dk"))s.dk=py::cast<double>(d["dk"]);
    if(d.contains("distance_step"))s.step=py::cast<double>(d["distance_step"]);
    if(d.contains("radial_order"))s.order=py::cast<int>(d["radial_order"]);
    if(d.contains("memory_limit_bytes"))s.budget=py::cast<I>(d["memory_limit_bytes"]);
    return s;
}
py::dict settings_dict(const Settings&s){py::dict d;d["kmax"]=s.kmax;d["dk"]=s.dk;d["distance_step"]=s.step;d["radial_order"]=s.order;d["memory_limit_bytes"]=s.budget;return d;}
void payload_budget(py::dict d,I budget) {
    I bytes=0;
    for(const char* key:{"radial_l","radial_species","grid_offsets","coeff_offsets","radius","coefficients"}) {
        require(py::isinstance<py::array>(d[key]),"payload must contain numerical arrays");
        auto a=py::cast<py::array>(d[key]);
        require(a.flags() & py::array::c_style,"payload arrays must be C contiguous");
        require(a.ndim()==1,"payload arrays must have one dimension");
        bool integral=std::string(key)!="radius" && std::string(key)!="coefficients";
        require(a.dtype().is(integral?py::dtype::of<I>():py::dtype::of<double>()),"payload dtype must be int64/float64");
        require(a.nbytes()<budget/4 && bytes<=budget/4-a.nbytes(),"radial payload exceeds memory budget before native copy");
        bytes+=a.nbytes();
    }
}
std::vector<Radial> read_payload(py::dict d){
    auto l=py::cast<Array<I>>(d["radial_l"]),species=py::cast<Array<I>>(d["radial_species"]),off=py::cast<Array<I>>(d["grid_offsets"]),coff=py::cast<Array<I>>(d["coeff_offsets"]);
    auto radius=py::cast<Array<double>>(d["radius"]),coeff=py::cast<Array<double>>(d["coefficients"]);
    I n=l.size();int ns=py::cast<int>(d["species_count"]);
    require(ns>0 && n>0 && species.size()==n && off.size()==n+1 && coff.size()==n+1,"bad radial payload shapes");
    require(off.data()[0]==0 && off.data()[n]==radius.size() && coff.data()[0]==0 && coff.data()[n]==coeff.size(),"invalid radial offsets");
    std::vector<Radial> out;
    for(I a=0;a<n;a++){
        I start=off.data()[a],end=off.data()[a+1],cs=coff.data()[a],ce=coff.data()[a+1];
        require(start>=0 && end<=radius.size() && end-start>=4 && cs>=0 && ce<=coeff.size() && ce-cs==4*(end-start-1),"bad radial grid/coefficients");
        require(l.data()[a]>=0 && l.data()[a]<=3,"supported input l range is 0..3");
        require(species.data()[a]>=0 && species.data()[a]<ns,"invalid radial species");
        Radial r{int(l.data()[a]),int(species.data()[a]),{radius.data()+start,radius.data()+end},{coeff.data()+cs,coeff.data()+ce}};
        require(r.radius.front()==0,"radial grid must start at zero");
        for(I j=0;j<I(r.radius.size());j++)require(std::isfinite(r.radius[j]) && (j==0||r.radius[j]>r.radius[j-1]),"radial grid not increasing");
        for(double v:r.coeff)require(std::isfinite(v),"nonfinite radial coefficient");out.push_back(std::move(r));
    }return out;
}
py::dict serialize(const Tables&t){
    py::dict d;d["settings"]=settings_dict(t.settings);d["nspecies"]=t.nspecies;d["ndistance"]=t.ndistance;d["step"]=t.step;
    std::vector<I>l,sp,off{0},coff{0},keys;std::vector<double>radius,coeff,values;
    for(auto&r:t.orbitals){l.push_back(r.l);sp.push_back(r.species);radius.insert(radius.end(),r.radius.begin(),r.radius.end());coeff.insert(coeff.end(),r.coeff.begin(),r.coeff.end());off.push_back(radius.size());coff.push_back(coeff.size());}
    py::dict p;p["radial_l"]=copy_array(l);p["radial_species"]=copy_array(sp);p["grid_offsets"]=copy_array(off);p["coeff_offsets"]=copy_array(coff);p["radius"]=copy_array(radius);p["coefficients"]=copy_array(coeff);p["species_count"]=t.nspecies;d["payload"]=p;
    for(auto&c:t.channels){for(int v:{c.a,c.b,c.lambda,c.power,c.L})keys.push_back(v);values.insert(values.end(),c.values.begin(),c.values.end());}
    d["channel_keys"]=copy_array(keys);d["values"]=copy_array(values);return d;
}
PYBIND11_MODULE(_operators_native,m){
    m.def("neighbors",[](Array<double> p,Array<double> c,Array<double> inv,Array<double> cut,I budget){
        require(p.ndim()==2 && p.shape(1)==3 && c.size()==9 && inv.size()==9 && cut.size()==p.shape(0),"invalid neighbor shapes");
        for(I j=0;j<p.size();j++)require(std::isfinite(p.data()[j]),"nonfinite position");
        for(I j=0;j<9;j++)require(std::isfinite(c.data()[j]) && std::isfinite(inv.data()[j]),"nonfinite cell");
        for(I j=0;j<cut.size();j++)require(std::isfinite(cut.data()[j]) && cut.data()[j]>0,"invalid cutoff");
        Neighbors a;{py::gil_scoped_release release;a=neighbors(p.data(),c.data(),inv.data(),cut.data(),p.shape(0),budget);}
        auto k=copy_array(a.keys);k.resize({py::ssize_t(a.keys.size()/5),py::ssize_t(5)});
        auto d=copy_array(a.displacement);d.resize({py::ssize_t(a.displacement.size()/3),py::ssize_t(3)});return py::make_tuple(k,d);
    });
    m.def("assemble_csr",[](Array<I> r,Array<I> c,Array<C> v,I n,I budget){
        require(r.ndim()==1 && c.ndim()==1 && v.ndim()==1 && r.size()==c.size() && r.size()==v.size(),"invalid COO shapes");
        CSR a;{py::gil_scoped_release release;a=csr(r.data(),c.data(),v.data(),r.size(),n,budget);}
        return py::make_tuple(copy_array(a.data),copy_array(a.indices),copy_array(a.indptr));
    });
    m.def("capabilities",[](){py::dict d;d["compiler"]=__VERSION__;d["openmp"]=true;d["openmp_version"]=_OPENMP;d["fast_math_available"]=true;d["algorithm"]="fourier-bessel-real-gaunt-v2-onsite";d["max_input_l"]=3;d["max_internal_l"]=7;return d;});
    py::class_<Tables>(m,"TableHandle").def("state",&serialize).def_property_readonly("settings",[](const Tables&t){return settings_dict(t.settings);}).def_property_readonly("channel_count",[](const Tables&t){return t.channels.size();});
    m.def("build_tables",[](py::dict payload,py::dict settings){auto s=read_settings(settings);payload_budget(payload,s.budget);auto radial=read_payload(payload);int ns=py::cast<int>(payload["species_count"]);py::gil_scoped_release release;return build_tables(std::move(radial),ns,s);});
    m.def("restore_tables",[](py::dict d){
        Tables t;t.settings=read_settings(py::cast<py::dict>(d["settings"]));payload_budget(py::cast<py::dict>(d["payload"]),t.settings.budget);t.orbitals=read_payload(py::cast<py::dict>(d["payload"]));t.nspecies=py::cast<int>(d["nspecies"]);t.ndistance=py::cast<int>(d["ndistance"]);t.step=py::cast<double>(d["step"]);
        require(t.nspecies>0 && t.nspecies==py::cast<int>(py::cast<py::dict>(d["payload"])["species_count"]),"restored species count mismatch");
        require(std::isfinite(t.step) && t.step>0 && t.step==t.settings.step,"invalid restored distance step");
        require(t.settings.budget>0 && t.settings.kmax>0 && t.settings.dk>0 && std::isfinite(t.settings.kmax+t.settings.dk) && t.settings.order>=2 && t.settings.order<=16,"invalid restored settings");
        std::set<int> species_ids;double maxcut=0;
        for(auto &r:t.orbitals){species_ids.insert(r.species);maxcut=std::max(maxcut,r.radius.back());}
        require(species_ids.size()==size_t(t.nspecies),"missing species in restored orbitals");
        require(t.ndistance>=6 && std::abs(t.step*(t.ndistance-1)-2*maxcut)<=1e-12*std::max(1.,maxcut),"restored distance grid does not cover support");
        auto keys=py::cast<Array<I>>(d["channel_keys"]);auto values=py::cast<Array<double>>(d["values"]);
        require(checked_product(values.size(),8)<t.settings.budget/4,"restored table exceeds memory budget");
        std::set<std::array<I,5>> expected,seen;
        for(I a=0;a<I(t.orbitals.size());a++)for(I b=0;b<I(t.orbitals.size());b++){
            int la=t.orbitals[a].l,lb=t.orbitals[b].l;
            for(int power=0;power<=1;power++)for(int lam=power?std::abs(la-1):la;lam<=la+power;lam+=2)
                for(int L=std::abs(lam-lb);L<=lam+lb;L+=2)expected.insert({a,b,lam,power,L});
        }
        require(t.ndistance>=6 && t.step>0 && keys.size()%5==0 && values.size()==keys.size()/5*t.ndistance,"invalid serialized table");
        for(I i=0;i<keys.size()/5;i++){
            auto k=keys.data()+5*i;
            std::array<I,5> channel{k[0],k[1],k[2],k[3],k[4]};
            require(expected.count(channel)==1 && seen.insert(channel).second,"invalid/duplicate restored channel");
            require(k[0]>=0 && k[1]>=0 && k[0]<I(t.orbitals.size()) && k[1]<I(t.orbitals.size()) && k[2]>=0 && k[2]<=4 && k[4]>=0 && k[4]<=7,"invalid table channel");
            t.channels.push_back({int(k[0]),int(k[1]),int(k[2]),int(k[3]),int(k[4]),{values.data()+i*t.ndistance,values.data()+(i+1)*t.ndistance}});
        }
        require(seen==expected,"restored table has missing channels");
        for(I i=0;i<values.size();i++)require(std::isfinite(values.data()[i]),"nonfinite table");
        {py::gil_scoped_release release;make_plans(t);}return t;
    });
    m.def("evaluate_pairs",[](const Tables&t,py::dict p,int threads){
        auto ids=py::cast<Array<I>>(p["pair_ids"]),sp=py::cast<Array<I>>(p["species"]);auto d=py::cast<Array<double>>(p["displacement_A"]);
        require(ids.ndim()==1 && sp.ndim()==2 && sp.shape(0)==ids.size() && sp.shape(1)==2 && d.ndim()==2 && d.shape(0)==ids.size() && d.shape(1)==3,"bad pair payload shape");
        std::vector<Pair>pairs;for(I i=0;i<ids.size();i++){require(sp.data()[2*i]>=0 && sp.data()[2*i]<t.nspecies && sp.data()[2*i+1]>=0 && sp.data()[2*i+1]<t.nspecies,"species index out of range");pairs.push_back({ids.data()[i],int(sp.data()[2*i]),int(sp.data()[2*i+1]),{d.data()[3*i],d.data()[3*i+1],d.data()[3*i+2]}});}
        Blocks b;{py::gil_scoped_release release;b=evaluate_pairs(t,pairs,threads);}
        py::dict out;out["pair_ids"]=copy_array(b.ids);out["offsets"]=copy_array(b.offsets);out["S"]=copy_array(b.s);auto D=copy_array(b.d);D.resize({py::ssize_t(3),py::ssize_t(b.s.size())});out["D"]=D;return out;
    },py::arg("table_handle"),py::arg("pair_payload"),py::arg("threads")=1);
}
