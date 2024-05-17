#include "itensor/all.h"
#include "spectralFun.h"
using namespace itensor;


int main()
    {
    int N = 10;
    double Jz = 1.0;
    double D = 1.0;

    std::string fname = "Axt_SmSp_N="+std::to_string(N)+"_Jz="+std::to_string(Jz)+"_D="+std::to_string(D)+".csv";

    // Make N spin 1 sites
    auto sites = SpinOne(N,{"ConserveQNs=",true});

    // Make the Hamiltonian for 1D Heisenberg
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

    //tdvp parameters
    auto t = 0.1;
    auto tend = 2.0;

    Args para;
    para.add("tstep", t);
    para.add("ttotal", tend);
    para.add("file_name",fname);

    spectralFunction(psi, H, energy, sites, "S-","S+", para);

    return 0;
    }
