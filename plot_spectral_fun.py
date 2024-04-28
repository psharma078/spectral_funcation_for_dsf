import sys
import numpy as np
from numpy import sin,cos,log,pi,sqrt,exp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import csv

num_cores = 16
fname=sys.argv[1]
Jz = 1
D=0
data = []
#loc = "/home/psharma/Spin_1_AFM_TEBD/SmSp_corr/"
#fname = loc+fname
with open(fname, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        row = np.array(row).astype("float")
        data.append(row)

data = np.array(data)
time = data[:,0]
site = data[:,1]
Axt = data[:,2]+1j*data[:,3]
Gxt = -1j*Axt
ReG = Gxt.real
ImG = Gxt.imag

#--------------------------------------------------------------------------
#real space-time profile of real & imaginary part of greens function G(x,t)
#--------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf=ax.plot_trisurf(time, site, ImG, linewidth=0.2, antialiased=True,cmap=plt.cm.YlGn)

ax.set_zlim(-2, 2)
ax.set_xlabel('time '+r'$(t)$',rotation=45,fontsize=12)
ax.set_ylabel('position '+r'$(x)$',fontsize=12)
ax.set_zlabel(r'$Im.G(x,t)$', rotation=180,fontsize=12)
ax.grid(False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elev=35, azim=45)
fig.savefig("ImG_N=100_Jz="+str(Jz)+"_D="+str(D)+".pdf",bbox_inches='tight')
plt.show()

#----------------------------------------------------
#----CALCULATING SPECTRAL FUNCTION-------------------
#---------------------------------------------------
L = 100
xo = int((L+1)/2)
delt=0.02
Tmax = max(time)
gaussian = exp(-4*(time/Tmax)**2)
ImG_gau = ImG*gaussian
site = site-xo

def dsf(q,omega):
    Gqw = 2*delt*np.sum(cos(omega*time)*cos(q*site)*ImG_gau)
    return -(1.0/np.pi)*Gqw

##~~~~~~ CUT ALONG q=pi ~~~~~~~~~~~~~~~~~~~~~~
fig1, ax1 = plt.subplots(figsize=(3,3))
omega = [0.01*n for n in range(301)]
k= pi
for w in omega:
    sqw = dsf(k,w)
    ax1.plot(w,sqw,marker='.',color='k')
ax1.axvline(x=0.416,alpha=0.7)
ax1.set_xlabel(r'$\omega$',fontsize=12)
ax1.set_ylabel(r'$S(q=\pi,\omega)$',fontsize=12)
fig1.savefig("dsf_cut_q=pi_N=100_Jz="+str(Jz)+"_D="+str(D)+".pdf",bbox_inches='tight')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Dynamical strcture factor color plot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
k = [2.0*np.pi/L*j for j in range(0,L+1)]
lenk = len(k)
omega = [0.01*n for n in range(301)]
lenw = len(omega)
k, omega = np.meshgrid(k,omega)
q_w = np.vstack((k.flatten(),omega.flatten())).T

solver = lambda j: dsf(q_w[j][0],q_w[j][1])
dyn_str_fac = np.array(Parallel(n_jobs=num_cores)(delayed(solver)(j) for j in range(len(q_w))))
dyn_str_fac = dyn_str_fac.reshape(lenw,lenk)

cmap = 'hot' #'rainbow'
fig2,ax2 = plt.subplots(figsize=(3.5,3))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
ax2.set_xlabel(r'$q$', weight='bold',fontsize=16)
ax2.set_ylabel(r'$\omega$', fontsize=16)

c = ax2.pcolormesh(k,omega,dyn_str_fac,shading = 'gouraud',cmap = cmap)
cb = fig2.colorbar(c, ax=ax2, shrink=1, pad=0.01, label=r'$S(q,\omega)$ [arb. unit]')
ax2.set_xlim(0,2.0*np.pi)
ax2.set_ylim(0,3)
fig2.savefig('sqw_N=100_Jz=1_D=0.pdf',bbox_inches='tight')
plt.show()
