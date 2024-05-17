#include "itensor/all.h"
#include "tdvp.h"
#include "basisextension.h"
using namespace itensor;

int main()
{
    int N = 10;
    double Jz = 1.0;
    double D = 1.0;

    // Make N spin 1 sites
    auto sites = SpinOne(N,{"ConserveQNs=",true});

    // Make the Hamiltonian MPO
    auto ampo = AutoMPO(sites);
    for(int i = 1; i < N; ++ i)
        {
        ampo += 0.5,"S+",i,"S-",i+1;
        ampo += 0.5,"S-",i,"S+",i+1;
        ampo += Jz, "Sz",i,"Sz",i+1;
        ampo += D, "Sz2", i;

        }
    ampo += D, "Sz2",N;
    auto H = toMPO(ampo);

    printfln("Maximum bond dimension of H is %d",maxLinkDim(H));

    // Set the initial state to be Neel state
    auto state = InitState(sites);
    for (int i:range1(N)) {state.set(i, i % 2 == 1 ? "Up" : "Dn");}

    auto phi = MPS(state);

    // Sweeping of DMRG
    auto DMRGsweeps = Sweeps(10);
    DMRGsweeps.maxm() = 10,20,50,100,100,200,200,200,500,500,500,1000,1000,1000,1000,1000,2000;
    DMRGsweeps.cutoff() = 1E-10;
    DMRGsweeps.niter() = 4;
    DMRGsweeps.noise() = 1E-6,1E-6,1E-7,1E-8,0,0,0,0;

    //Begin DMRG calculation
    auto [energy,psi] = dmrg(H,phi,DMRGsweeps,"Quiet");

    printfln("Ground state energy = ",energy);
    println("\nTotal QN of Ground State = ",totalQN(psi));
    println("\noverlap with the initial state = ",inner(phi,psi));

    //----------- Begin spectral function A(x,t) calculation --------------
    int central_site = (N+1)/2;
    auto psi0 = psi; //copy the ground state MPS
    auto psi0dag = dag(psi0);
    auto sp_psi = op(sites,"S+",central_site)*psi(central_site);
    sp_psi.noPrime();
    psi.set(central_site,sp_psi); //replace middle site with new MPS block

    // start TDVP, either one site or two site algorithm can be used by adjusting the "NumCenter" argument
    println("-----------------------GSE-TDVP-----------------------------");

    auto sweeps = Sweeps(1);
    sweeps.maxdim() = 2000;
    sweeps.cutoff() = 1E-12;
    sweeps.niter() = 10;

    auto t = 0.1;
    auto tend = 1.0;
    int nsw = tend/t;
    auto Energy = 0.0;

  for(int n = 1; n <= nsw; ++n)
    {
        if(n < 5)
            {
            // Global subspace expansion
            std::vector<Real> epsilonK = {1E-12, 1E-12};
            addBasis(psi,H,epsilonK,{"Cutoff",1E-8,
                                      "Method","DensityMatrix",
                                      "KrylovOrd",3,
                                      "DoNormalize",false,
                                      "Quiet",true});
            }

        // TDVP sweep
        Energy = tdvp(psi,H,-Cplx_i*t,sweeps,{"Truncate",true,
                                        "DoNormalize",false,
                                        "Quiet",true,
                                        "NumCenter",1,
                                        "ErrGoal",1E-7});

        printfln("\nEnergy after imaginary time = %.10f is e = %.10f",t*n, Energy);//shouldn't change for real-time evolution (unitary operator)

    for (int i=1;i<=N;++i)
        {
        auto Sm = op(sites,"S-",i);
        auto C = psi(1);
        if (i==1){C *= Sm*prime(psi0dag(i),"Site");}
        else {C *= psi0dag(1);}

        for (int k=2;k<=N;++k)
        {
        if (k==i){
         C *= psi(k);
         C *= Sm*prime(psi0dag(k),"Site");}
        else {
         C *= psi(k);
         C*= psi0dag(k);}
        }
        auto result = exp(Cplx_i*energy*t*n)*eltC(C);
        printfln("",n*t," ", i, " ", result);
        }
    }

 return 0;
}
