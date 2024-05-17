#pragma once
#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include "tdvp.h"
#include "basisextension.h"
#include <fstream>

using namespace itensor;
using std::vector;

void collectdata(float t, int x, float RG, float ImG, const std::string& fname) {
    std::ofstream outfile(fname, std::ios::app);
    outfile << t << "," << x << "," << RG << "," << ImG << "\n";
    outfile.close();
}

// Function for computing spectral function A(x,t)
void spectralFunction(MPS psi, MPO& H, Real& energy, const SiteSet& sites, const std::string& opr1,
                      const std::string& opr2, const Args& param) {
    auto N = length(psi);

    auto file_name = param.getString("file_name", "filename");

    auto corr = correlationMatrix(psi, sites,opr1,opr2);
    //ground state correlation
    cout<<endl;
    cout<<"i, j,  "<< opr1 << opr2 << endl;
    for (auto i=0; i<N;++i)
    {
        for (auto j=0; j<N;++j)
        {
         if (j<=i) printfln("",i+1," ", j+1, " ", corr[i][j]);
         if (j+1==(N+1)/2) collectdata(0.0, i+1, corr[i][j], 0.0, file_name);
        }
    }

    //----------- Begin spectral function A(x,t) calculation --------------
    int central_site = (N + 1) / 2;
    auto psi0 = psi; // Copy the ground state MPS
    auto psi0dag = dag(psi0);
    auto sp_psi = op(sites, opr2, central_site) * psi(central_site);
    sp_psi.noPrime();
    psi.set(central_site, sp_psi); // Replace middle site with new MPS block

    // Start TDVP, either one site or two-site algorithm can be used by adjusting the "NumCenter" argument
    println("-----------------------GSE-TDVP-----------------------------");

    auto sweeps = Sweeps(1);
    sweeps.maxdim() = 2000;
    sweeps.cutoff() = 1E-12;
    sweeps.niter() = 10;

    auto t = param.getReal("tstep", 0.0);
    auto tend = param.getReal("ttotal", 0.0);
    int nsw = tend / t;

    auto Energy = 0.0; // Initialize Energy for tdvp step
    for (int n = 1; n <= nsw; ++n) {
        if (n < 3) {
            // Global subspace expansion
            std::vector<Real> epsilonK = {1E-12, 1E-12};
            addBasis(psi, H, epsilonK, {"Cutoff", 1E-8,
                                        "Method", "DensityMatrix",
                                        "KrylovOrd", 3,
                                        "DoNormalize", false,
                                        "Quiet", true});
        }
        
        // TDVP sweep, real time evolution
        Energy = tdvp(psi, H, -Cplx_i * t, sweeps, {"Truncate", true,
                                                    "DoNormalize", false,
                                                    "Quiet", true,
                                                    "NumCenter", 1,
                                                    "ErrGoal", 1E-7});

        printfln("\nEnergy after real time t = %.10f is energy = %.10f", t * n, Energy);

        for (int i = 1; i <= N; ++i) {
            auto Sm = op(sites, opr1, i);
            auto C = psi(1);
            if (i == 1) { C *= Sm * prime(psi0dag(i), "Site"); }
            else { C *= psi0dag(1); }

            for (int k = 2; k <= N; ++k) {
                if (k == i) {
                    C *= psi(k);
                    C *= Sm * prime(psi0dag(k), "Site");
                }
                else {
                    C *= psi(k);
                    C *= psi0dag(k);
                }
            }
            auto result = exp(Cplx_i * energy * t * n) * eltC(C);
            collectdata(t * n, i, result.real(), result.imag(), file_name);
        }
    }
} // spectralFunction
