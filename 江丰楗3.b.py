import numpy as np
import matplotlib.pyplot as plt
import copy
from pylab import quiver

def get_next_vor_int(dt,h,vis,v1,s):

    #this function is to get next vorticity in the interior
    #input: dt = timestep, h = distance between grid point, vis = viscosity
    #input: v1 = temporary vorticity everywhere (matters in the process), s = temporary streamfunction everywhere
    
    Nx,Ny = v1.shape[0],v1.shape[1]
    v2 = copy.deepcopy(v1)
    
    for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                v2[i,j] = v1[i,j]+dt*(((s[i+1,j]-s[i-1,j])*(v1[i,j+1]-v1[i,j-1])/(4*h**2))-\
                                      ((s[i,j+1]-s[i,j-1])*(v1[i+1,j]-v1[i-1,j])/(4*h**2))+\
                                      vis*(v1[i+1,j]+v1[i-1,j]+v1[i,j+1]+v1[i,j-1]-4*v1[i,j])/(h**2))
                
    #output: v2 = next vorticity (only interior matters)
    return v2


def get_next_strf(s,v,h,maxIter=100,beta=1.5,maxErr=0.001):
    
    #this function is to get next streamfunction everywhere through solving poisson equation
    #input: s = last streamfunction to be updated,v = next vorticity (only interior matters), h = distance between grid point
    
    Nx,Ny = v.shape[0],v.shape[1]
    temp = np.zeros((Nx,Ny))

    # update s[x,y] inside domain
    for iters in range(maxIter):   #SOR iteration
        temp = copy.deepcopy(s)
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                s[i,j]=0.25*beta*(s[i+1,j]+s[i-1,j]+s[i,j+1]+s[i,j-1]+h*h*v[i,j])+\
                         (1-beta)*s[i,j]
                
                
        err = sum(sum(np.abs(s-temp)))
        if err<=maxErr:
            break

    #output: s = next streamfunction (complete)
    return s

                
def get_next_vor_bnd(h,U,v,s):
    
    #this function it to get next vorticity at the boundary
    #input: h = distance between grid point, U = velocity at the wall(top, constant)
    #input: v = next vorticity (only interior matters), s = next streamfunction everywhere
    v[:,0] = -2*s[:,1]/(h**2)
    v[:,-1] =-2*s[:,-2]/(h**2)
    v[-1,:] =-2*s[-2,:]/(h**2)-2*U/h
    v[0,:] = -2*s[1,:]/(h**2)
                                
    #output: v = next vorticity everwhere (boundary updated in this function)
    return v

#define vorticity and streamfunciton as 2-D arrays
vor = np.zeros((17,17))
strf = np.zeros((17,17))

#parameters
dt = 0.005
h = 1.0/16
vis = 0.1
U = 1.0
x = np.linspace(0,1,17)
y = np.linspace(0,1,17)
X,Y = np.meshgrid(x,y)
n = 200

time = np.arange(0,1+dt,dt)
Q = np.zeros(n+1)

#update vorticity and streamfunction through each timestep
for i in range(n):
    Q[i] = -strf.min()
    result1 = get_next_vor_int(dt,h,vis,vor,strf)
    result2 = get_next_strf(strf,result1,h)
    result3 = get_next_vor_bnd(h,U,result1,result2)
    vor = result3.copy()
    strf = result2.copy()
    
Q[n] = -strf.min()    
plt.contour(X,Y,vor,40)
plt.axis('square')
plt.show()

plt.contour(X,Y,strf,10)
plt.axis('square')
plt.show()

plt.figure()
plt.plot(time,Q,color='black')
plt.xlim(0,1)
plt.ylim(0,0.1)
plt.show()

u = np.zeros((17,17))
v = np.zeros((17,17))

for k in range(17):
    u[-1,k] = U

for i in range(1,16):
    for j in range(1,16):
        u[i,j] = (strf[i+1,j]-strf[i-1,j])/(2*h)
        v[i,j] = -(strf[i,j+1]-strf[i,j-1])/(2*h)

plt.figure()
quiver(X,Y,u,v,color='purple')
plt.axis('square')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()