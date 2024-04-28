#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <fstream>
#include <iostream>
#include "spinCorrelation.h"
using namespace itensor;
using std::vector;

void collectdata(float t, int x,float RG,float ImG, std::string fname){
        std::fstream outfile(fname,std::ios::app | std::ios::in);
//      std::fstream outfile("try.txt",std::ios::app | std::ios::in);
//             time,  position Re.Green fun, Im(Green_fun)
        outfile<<t<<","<<x<<","<<RG<<","<<ImG<<"\n";
        outfile.close();
}

int
main(int argc, char* argv[])
{
int N = 100;  //atoi(argv[1]);
double Jz = atof(argv[1]);
double D = atof(argv[2]);
printfln("N= ",N);
printfln("Jz= ", Jz);
printfln("D= ",D);

std::string file_name = "Axt_SmSp_N=100_Jz="+std::to_string(Jz)+"_D="+std::to_string(D)+"_run2.csv";

auto sites=SpinOne(N,{"ConserveQNs=",true});

//nearest-neighbor Heisenberg model
auto ampo = AutoMPO(sites);
for (int j=1; j<N;j++)
  {
  ampo += 0.5,"S+",j,"S-",j+1;
  ampo += 0.5,"S-",j,"S+",j+1;
  ampo += Jz,"Sz",j,"Sz",j+1;
  ampo += D,"Sz2",j;
  }
ampo += D,"Sz2",N;
//ampo += 0.5,"S+",1,"S-",N;
//ampo += 0.5,"S-",1,"S+",N;
//ampo += Jz,"Sz",1,"Sz",N;

auto H = toMPO(ampo);

//set initial wavefunction matrix product state
auto state = InitState(sites);

for (auto j : range1(N))
{
state.set(j,j%2==1?"Up":"Dn");  //Neel state
}
auto phi = MPS(state);

    // Sweeping of DMRG
    auto sweeps = Sweeps(60);
    sweeps.maxdim() = 10,20,50,100,100,200,200,200,500,500,500,1000,1000,1000,1000,1000,2000;
    //sweeps.maxm() = 100,200,200,500,500,500,1000,1000,2000,2000,2000,3000,3000,3000;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 4;
    sweeps.noise() = 1E-6,1E-6,1E-7,1E-8,0,0,0,0;
    //println(sweeps);

//Begin DMRG calculation
auto [energy,psi] = dmrg(H,phi,sweeps,"Quiet");

printfln("Ground state energy = ",energy);
println("\nTotal QN of Ground State = ",totalQN(psi));
println("\noverlap with the initial state = ",inner(phi,psi));

std::cout<<"magnetization"<<std::endl;
std::cout<<"i <Sz>"<<std::endl;
for (int i : range1(N))
    {
        auto sz = sites.op("Sz",i);
        psi.position(i);
        auto C = psi(i)*sz;
        C *= dag(prime(psi(i),"Site"));
        std::cout<<i<<" "<<elt(C)<<std::endl;
    }

std::cout<<std::endl;
spinCorrelator(psi,sites);

Real tstep = 0.02; //time step (smaller is generally more accurate)
//Real ttotal = 1.0; //total time to evolve
Real cutoff = 1E-8; //truncation error cutoff when restoring MPS form

//Create a std::vector (dynamically sizeable array)
//to hold the Trotter gates
auto gates = vector<BondGate>();

//Create the gates exp(-i*tstep/2*hterm)
//and add them to gates
for(int b = 1; b <= N-1; ++b)
    {
    auto hterm = Jz*op(sites,"Sz",b)*op(sites,"Sz",b+1);
    hterm += 0.5*op(sites,"S+",b)*op(sites,"S-",b+1);
    hterm += 0.5*op(sites,"S-",b)*op(sites,"S+",b+1);

    auto g = BondGate(sites,b,b+1,BondGate::tReal,tstep/2.,hterm);
    gates.push_back(g);
    }
//Create the gates exp(-i*tstep/2*hterm) in reverse
//order (to get a second order Trotter breakup which
//does a time step of "tstep") and add them to gates
    for(int b = N-1; b >= 1; --b)
    {
    auto hterm = Jz*op(sites,"Sz",b)*op(sites,"Sz",b+1);
    hterm += 0.5*op(sites,"S+",b)*op(sites,"S-",b+1);
    hterm += 0.5*op(sites,"S-",b)*op(sites,"S+",b+1);

    auto g = BondGate(sites,b,b+1,BondGate::tReal,tstep/2.,hterm);
    gates.push_back(g);
    }

    //for onsite terms use the identity operator to make it two site operator
    for (int b =1; b<=N-1; ++b)
    {
    auto hterm = D*op(sites,"Sz2",b)*op(sites,"Id",b+1);  //forward gate
    //hterm += -B*op(sites,"Sz",b)*op(sites,"Id",b+1);
    hterm += D*op(sites,"Sz2",b+1)*op(sites,"Id",b);  //reverse gate
    //hterm += -B*op(sites,"Sz",b+1)*op(sites,"Id",b);

    if (b==1) //for boundary,as for boundary, gates will be applied only once per site
    {
    hterm += D*op(sites,"Id",b+1)*op(sites,"Sz2",b);
    //hterm += -B*op(sites,"Id",b+1)*op(sites,"Sz",b);
    }
    if (b==N-1)
    {
    hterm += D*op(sites,"Id",b)*op(sites,"Sz2",b+1);
    //hterm += -B*op(sites,"Id",b)*op(sites,"Sz",b+1);
    }
    auto g = BondGate(sites,b,b+1,BondGate::tReal,tstep/2.,hterm);
    gates.push_back(g);
    }
//Save initial state;
auto psi0 = psi;
auto psi0dag = dag(psi0);
auto sp_psi = op(sites,"S+",(N+1)/2)*psi((N+1)/2);
sp_psi.noPrime();
psi.set((N+1)/2,sp_psi); //replace middle site with new MPS block

for (int t = 1; t<=1200; ++t)
{
gateTEvol(gates,tstep,tstep,psi,{"Cutoff=",cutoff,"Verbose=",true,"MaxDim",300,"Normalize",false});
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
         C *= Sm*prime(psi0dag(k),"Site");
         }
 else {
      C *= psi(k);
      C*= psi0dag(k);
      }
 }
auto result = exp(Cplx_i*energy*t*tstep)*eltC(C);
//auto result = -Cplx_i*sqrt(2)*exp(Cplx_i*energy*t*tstep)*eltC(C);
collectdata(t*tstep,i,result.real(),result.imag(), file_name);

}
printfln("Maximum MPS bond dimension after time evolution is %d",maxLinkDim(psi));
}

return 0;
}
