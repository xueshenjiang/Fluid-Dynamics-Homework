import numpy as np
import matplotlib.pyplot as plt
import copy

def solve_poisson(f,s,h,maxIter=100,beta=1.5,maxErr=0.001):
    
    #this function is to solve 2-D poisson equation (Laplacian equation when s=0) 
    #input: f = f(x,y) to be solved, s = source function s(x,y), h = distance between grid point
    
    Nx,Ny = s.shape[0],s.shape[1]
    temp = np.zeros((Nx,Ny))

    # update f(x,y) inside domain
    for iters in range(maxIter):   #SOR iteration
        temp = copy.deepcopy(f)
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                f[i,j]=0.25*beta*(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1]-h*h*s[i,j])+\
                         (1-beta)*f[i,j]
                
                
        err = sum(sum(np.abs(f-temp)))
        if err<=maxErr:
            break

    #output: f = result f(x,y)
    return f    


x = np.linspace(0,1,17)
y = np.linspace(0,1,17)
X,Y = np.meshgrid(x,y)
h = 1/16

#first we solve a Laplacian equation with non-zero boundary condition
f1 = np.zeros((17,17))
s1 = np.zeros((17,17))
sinbc = np.sin(2*np.pi*x)
f1[0,:] = sinbc
#plot our numerical solution res1
res1 = solve_poisson(f1,s1,h)
plt.figure()
ax1 = plt.axes(projection ='3d')
ax1.plot_surface(X,Y,res1,rstride=1,cstride=1,cmap='rainbow')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('res1')
plt.show()
#plot our analitical solution anal1
anal1 = (np.exp(4*np.pi**2-2*np.pi*Y)-np.exp(2*np.pi*Y))*np.sin(2*np.pi*X)/(np.exp(4*np.pi**2)-1)
plt.figure()
ax3 = plt.axes(projection ='3d')
ax3.plot_surface(X,Y,anal1,rstride=1,cstride=1,cmap='rainbow')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('anal1')


#next we solve a Poisson equation with center source s(x,y)
f2 = np.zeros((17,17))
s2 = np.zeros((17,17))
s2 = -np.sin(np.pi*X)*np.sin(np.pi*Y)
res2 = solve_poisson(f2,s2,h)
#plot our numerical solution res2
plt.figure()
ax2 = plt.axes(projection ='3d')
ax2.plot_surface(X,Y,res2,rstride=1,cstride=1,cmap='rainbow')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('res2')
plt.show()
#plot our analitical solution anal2
anal2 = np.sin(np.pi*X)*np.sin(np.pi*Y)/(2*np.pi**2)
plt.figure()
ax4 = plt.axes(projection ='3d')
ax4.plot_surface(X,Y,anal2,rstride=1,cstride=1,cmap='rainbow')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('anal2')