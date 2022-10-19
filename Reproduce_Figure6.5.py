import numpy as np
import matplotlib.pyplot as plt


#one-dimensional advection-diffusion by the FTCS scheme

D = 0.05
U = 1
time = 0.5
space = 2.0

dt_1 = 0.05
dx_1 = 0.1
a_1 = (D*dt_1)/(dx_1**2)-(U*dt_1)/(2*dx_1)
b_1 = 1-(2*D*dt_1)/(dx_1**2)
c_1 = (D*dt_1)/(dx_1**2)+(U*dt_1)/(2*dx_1)

Nt_1 = 11 #=time/dt_1
Nx_1 = 21 #=space/dx_1

xarr_1 = np.linspace(0,2.0,Nx_1)
xarr_anal = np.linspace(0,2.0,2001)
numsoln_1 = np.zeros((Nt_1,Nx_1))
analsoln = np.exp(-0.5*0.05*4*(np.pi**2))*np.sin(2*np.pi*(xarr_anal-0.5))

#inicial condition for numsoln_1
numsoln_1[0][:] = np.sin(2*np.pi*xarr_1)

for i in range(1,Nt_1):
    for j in range(1,Nx_1-1):
        numsoln_1[i][j] = a_1*numsoln_1[i-1][j+1]+b_1*numsoln_1[i-1][j]+c_1*numsoln_1[i-1][j-1]
    numsoln_1[i][Nx_1-1] = a_1*numsoln_1[i-1][1]+b_1*numsoln_1[i-1][Nx_1-1]+c_1*numsoln_1[i-1][Nx_1-2]
    numsoln_1[i][0] = numsoln_1[i][Nx_1-1]
    
plt.figure()
plt.plot(xarr_anal,analsoln,label = 'Exact',color = 'red')
plt.plot(xarr_1,numsoln_1[Nt_1-1][:],label = 'Numerical',color = 'black')
plt.xlim(0,2)
plt.ylim(-1.5,1.5)
plt.xticks(np.arange(0,2.5,0.5))
plt.yticks(np.arange(-1.5,2.0,0.5))
plt.text(0.1,1.3,"m=21;time=0.50")
plt.legend(loc = 'lower left',frameon = False)
plt.show()

dt_2 = 0.0005
dx_2 = 0.01
a_2 = (D*dt_2)/(dx_2**2)-(U*dt_2)/(2*dx_2)
b_2 = 1-(2*D*dt_2)/(dx_2**2)
c_2 = (D*dt_2)/(dx_2**2)+(U*dt_2)/(2*dx_2)

Nt_2 = 1001 #=time/dt_1+1
Nx_2 = 201 #=space/dx_1+1

xarr_2 = np.linspace(0,2.0,Nx_2)
numsoln_2 = np.zeros((Nt_2,Nx_2))


#inicial condition for numsoln_2
numsoln_2[0][:] = np.sin(2*np.pi*xarr_2)

for i in range(1,Nt_2):
    for j in range(1,Nx_2-1):
        numsoln_2[i][j] = a_2*numsoln_2[i-1][j+1]+b_2*numsoln_2[i-1][j]+c_2*numsoln_2[i-1][j-1]
    numsoln_2[i][Nx_2-1] = a_2*numsoln_2[i-1][1]+b_2*numsoln_2[i-1][Nx_2-1]+c_2*numsoln_2[i-1][Nx_2-2]
    numsoln_2[i][0] = numsoln_2[i][Nx_2-1]
    
plt.figure()
plt.plot(xarr_anal,analsoln,label = 'Exact',color = 'red',linewidth = 3)
plt.plot(xarr_2,numsoln_2[Nt_2-1][:],label = 'Numerical',color = 'black')
plt.xlim(0,2)
plt.ylim(-1.5,1.5)
plt.xticks(np.arange(0,2.5,0.5))
plt.yticks(np.arange(-1.5,2.0,0.5))
plt.legend(loc = "best")
plt.text(0.1,1.3,"m=201;time=0.50")
plt.legend(loc = 'lower left',frameon = False)
plt.show()