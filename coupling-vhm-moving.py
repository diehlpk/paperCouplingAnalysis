#!/usr/bin/env python3
# Coupling using a variable horizon (VHCM)
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 02/05/2021
import numpy as np
import sys 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter


pgf_with_latex = {"text.usetex": True, "font.size" : 12, "pgf.preamble" : [r'\usepackage{xfrac}'] }


example = sys.argv[1]
solution = sys.argv[2]

g = -1 
eps = 0.01
condition = False


#############################################################################
# Solve the system
#############################################################################

def solve(M,f):
    return np.linalg.solve(M,f)

#############################################################################
# Loading
#############################################################################

def f(x):
    
    global g 

    if example == "Cubic":
        g = 27
        return -6*x
    elif example == "Quartic":
        g = 108
        return -12 * x*x
    elif example == "Quadratic":
        g = 6
        return -2
    elif example == "Linear":
        g = 1
        return 0
    elif example == "Linear-cubic":
        g = 31./4.
        if x < 1.5:
            return 0 
        else:
            return 9-6*x
    elif example == "Sin":
        g = 2*np.pi*np.cos(2*np.pi*3)
        return 4*np.pi*np.pi*np.sin(2*np.pi*x)
    elif example == "Cos":
        g = -2*np.pi*np.sin(2*np.pi*3+np.pi/2)
        return 4*np.pi*np.pi*np.cos(2*np.pi*x+np.pi/2)
    elif example == "Steep" :
        g = 1-np.exp((3/3-1)/eps)/(eps*(1-np.exp(-1/eps)))
        return np.exp((x/3-1)/eps)/(3*eps*eps*(1-np.exp(-1/eps)))
    elif example == "Tan":
        g = (np.pi * np.power(np.sin(np.pi * 3 /8),2)) / ( 8 * np.power(np.cos( np.pi * 3 / 8),2)) + np.pi / 8
        return -((np.pi * np.pi * np.power(np.sin(np.pi * x / 8),3)) / (32 * np.power(np.cos( np.pi * x / 8),3)) + (np.pi * np.pi * np.power(np.sin(np.pi * x / 8),1)) / (32 * np.cos( np.pi * x /8)))
    else:
        print("Error: Either provide Linear, Quadratic, Quartic, or Cubic")
        sys.exit()

def forceFull(n,h):
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(i * h)
    
    force[n-1] = 0 #g
    
    return force

def forceCoupling(n,x):
    
    force = np.zeros(n)
    
    for i in range(1,n-1):
        force[i] = f(x[i])
    
    force[n-1] = 0 #g
    
    return force

#############################################################################
# Exact solution 
#############################################################################

def exactSolution(x):
    
    if example == "Cubic":
        return x * x * x
    elif example == "Quartic":
        return x * x * x * x
    elif example == "Quadratic":
        return x * x
    elif example == "Linear":
        return x
    elif example == "Linear-cubic":
        return np.where(x < 1.5, x, x + (x-1.5) * (x-1.5) * (x-1.5) )
    elif example == "Sin":
        return np.sin(2*np.pi*x)
    elif example == "Cos":
       return np.cos(2*np.pi*x+np.pi/2) 
    elif example == "Steep":
        return x-3*(np.exp((x/3-1)/eps)-np.exp(-1/eps))/(1-np.exp(-1/eps))
    elif example == "Tan":
        return np.tan(3*x/4*np.pi/6)
    else:
        print("Error: Either provide Linear, Quadratic, Quartic, or Cubic")
        sys.exit()

#############################################################################
# Assemble the stiffness matrix for the finite difference model (FD)
#############################################################################

def FDM(n,h):

    M = np.zeros([n,n])

    M[0][0] = 1

    for i in range(1,n-1):
        M[i][i-1] = -2
        M[i][i] = 4
        M[i][i+1] = -2

    M[n-1][n-1] = 1
    #M[n-1][n-1] = 11*h / 3 
    #M[n-1][n-2] = -18*h / 3
    #M[n-1][n-3] = 9* h / 3
    #M[n-1][n-4] = -2* h / 3


    M *= 1./(2.*h*h)

    return M

#############################################################################
# Assemble the stiffness matrix for the coupling of FDM - FDM - FDM
#############################################################################

def CouplingFDFD(n,h):

    M = np.zeros([3*n,3*n])

    M[0][0] = 1

    for i in range(1,n-1):
        M[i][i-1] = -2
        M[i][i] = 4
        M[i][i+1] = -2

    M[n-1][n-1] = -1 
    M[n-1][n] = 1 

    M[n][n-1] = 3*h
    M[n][n-2] = -4*h
    M[n][n-3] = 1*h

    M[n][n] = 3*h
    M[n][n+1] = -4*h
    M[n][n+2] = 1*h

    for i in range(n+1,2*n-1):
        M[i][i-1] = -2
        M[i][i] = 4
        M[i][i+1] = -2

    M[2*n-1][2*n-1] = -1 
    M[2*n-1][2*n] = 1

    M[2*n][2*n-1] = 3*h
    M[2*n][2*n-2] = -4*h
    M[2*n][2*n-3] = h

    M[2*n][2*n] = 3*h
    M[2*n][2*n+1] = -4*h
    M[2*n][2*n+2] = h

    for i in range(2*n+1,3*n-1):
        M[i][i-1] = -2
        M[i][i] = 4
        M[i][i+1] = -2

    M[3*n-1][3*n-1] = 11*h / 3
    M[3*n-1][3*n-2] = -18*h / 3
    M[3*n-1][3*n-3] = 9 * h / 3
    M[3*n-1][3*n-4] = -2 * h / 3


    M *= 1./(2.*h*h)

    return M

#############################################################################
# Assemble the stiffness matrix for the varibale horizon model (VHM)
#############################################################################

def VHM(n,h):
        
    MVHM = np.zeros([n,n])

    MVHM[0][0] = 1.

    MVHM[1][0] = -8.
    MVHM[1][1] = 16.
    MVHM[1][2] = -8.

    for i in range(2,n-2):
        MVHM[i][i-2] = -1.
        MVHM[i][i-1] = -4.
        MVHM[i][i] = 10.
        MVHM[i][i+1] = -4.
        MVHM[i][i+2] = -1.


    MVHM[n-2][n-1] = -8.
    MVHM[n-2][n-2] = 16.
    MVHM[n-2][n-3] = -8.

    MVHM[n-1][n-1] = 12.*h
    MVHM[n-1][n-2] = -16.*h
    MVHM[n-1][n-3] = 4.*h

    MVHM *= 1./(8.*h*h)
    
    return  MVHM


#############################################################################
# Assemble the stiffness matrix for the coupling of FDM - VHM - FDM
#############################################################################

def CouplingFDVHM(nodes1,nodes2,nodes3,h):

    total = nodes1 + nodes2 + nodes3

    fVHM = 1./(8.*h*h)
    fFDM = 1./(2.*h*h)

    M = np.zeros([total,total])
    
    M[0][0] = 1 * fFDM

    for i in range(1,nodes1-1):
        M[i][i-1] = -2 * fFDM
        M[i][i] = 4 * fFDM
        M[i][i+1] = -2 * fFDM

    M[nodes1-1][nodes1-1] = -1 
    M[nodes1-1][nodes1] = 1   

    M[nodes1][nodes1-1] = 11*h * fFDM / 3
    M[nodes1][nodes1-2] = -18*h * fFDM / 3
    M[nodes1][nodes1-3] = 9*h * fFDM / 3
    M[nodes1][nodes1-4] = -2*h * fFDM / 3

    M[nodes1][nodes1] = 11 / 6 / h 
    M[nodes1][nodes1+1] = -18 / 6 / h
    M[nodes1][nodes1+2] = 9 / 6 / h
    M[nodes1][nodes1+3] = -2 / 6 / h


    M[nodes1+1][nodes1] = -8 * fVHM
    M[nodes1+1][nodes1+1] = 16 * fVHM
    M[nodes1+1][nodes1+2] = -8 * fVHM

    for i in range(nodes1+2,nodes1+nodes2-2):
        M[i][i-2] = -1. * fVHM
        M[i][i-1] = -4. * fVHM
        M[i][i] = 10. * fVHM
        M[i][i+1] =  -4. * fVHM
        M[i][i+2] = -1. * fVHM

    n = nodes1 + nodes2

    M[n-2][n-3] = -8 * fVHM
    M[n-2][n-2] = 16 * fVHM
    M[n-2][n-1] = -8 * fVHM

    M[n-1][n-1] = -1 
    M[n-1][n] = 1  

    M[n][n-1] = 11 / 6 / h
    M[n][n-2] = -18 / 6 / h
    M[n][n-3] = 9 / 6 / h
    M[n][n-4] = -2 / 6 / h

    M[n][n] = 11*h * fFDM / 3
    M[n][n+1] = -18*h * fFDM / 3
    M[n][n+2] = 9*h * fFDM / 3
    M[n][n+3] = -2*h * fFDM / 3


    for i in range(n+1,n+nodes3-1):
        M[i][i-1] = -2 * fFDM
        M[i][i] = 4 * fFDM
        M[i][i+1] = -2 * fFDM

    n += nodes3
    
    M[n-1][n-1] = 1 
    #M[n-1][n-1] = 11*h * fFDM / 3
    #M[n-1][n-2] = -18*h * fFDM / 3 
    #M[n-1][n-3] = 9 * h * fFDM / 3
    #M[n-1][n-4] = -2 * h * fFDM / 3

    if condition :
        print(np.linalg.cond(M))
        with open("con-vhm-neumann.txt", "a") as f:
            f.write(str(np.linalg.cond(M))+"\n")
            f.close()
            
    return M

markers = ['s','o','x','.']

start = 4
for i in range(start,start+4):
    n = np.power(2,i)
    h = 1./n
    nodes1 = int(0.5/h)+1
    nodes2 = int(1.75/h)+1
    nodes3 = int(0.75/h)+1
    nodesFull = 3 * n + 1

    print(nodesFull,h)
    x1 = np.linspace(0,0.5,nodes1)
    x2 = np.linspace(0.5,2.25,nodes2)
    x3 = np.linspace(2.25,3.,nodes3)
    x = np.array(np.concatenate((x1,x2,x3)))

    xFull = np.linspace(0,3.,nodesFull)

    forceCoupled = forceCoupling(nodes1+nodes2+nodes3,x)

    forceCoupled[nodes1-1] = 0
    forceCoupled[nodes1] = 0

    forceCoupled[nodes1+nodes2-1] = 0
    forceCoupled[nodes1+nodes2] = 0

    uFDMVHM = solve(CouplingFDVHM(nodes1,nodes2,nodes3,h),forceCoupled)
    uSlice = np.array(np.concatenate((uFDMVHM[0:nodes1],uFDMVHM[nodes1+1:nodes1+nodes2],uFDMVHM[nodes1+nodes2+1:nodes1+nodes2+nodes3])))

    uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))

    print("------")
    #print(uFDMVHM[nodes1-1])
    #print(uFDMVHM[nodes1])
    #print(uFD[nodes1-1])
    print("PD")
    upfd = (-2 * uFDMVHM[nodes1-4]   + 9 *  uFDMVHM[nodes1-3]  - 18 * uFDMVHM[nodes1-2] +  11 * uFDMVHM[nodes1-1] ) / 6
    uppd = (-2 * uFDMVHM[nodes1+3]   + 9 *  uFDMVHM[nodes1+2]  - 18 * uFDMVHM[nodes1+1] +  11 * uFDMVHM[nodes1] ) / 6
    print(upfd)
    print(uppd)
    print(abs(uppd)-abs(upfd))

    upfdl = (-2 * uFD[nodes1-4]   + 9 *  uFD[nodes1-3]  - 18 * uFD[nodes1-2] +  11 * uFD[nodes1-1] ) / 6
    upfdr = (-2 * uFD[nodes1+2]   + 9 *  uFD[nodes1+1]  - 18 * uFD[nodes1] +  11 * uFD[nodes1-1] ) / 6
    print("FDM ")
    print(upfdl)
    print(upfdr)
    print(abs(upfdl)-abs(upfdr))

    print("-- Second interface")
    upfd = (-2 * uFDMVHM[nodes1+nodes2-4]   + 9 *  uFDMVHM[nodes1+nodes2-3]  - 18 * uFDMVHM[nodes1+nodes2-2] +  11 * uFDMVHM[nodes1+nodes2-1] ) / 6
    uppd = (-2 * uFDMVHM[nodes1+nodes2+3]   + 9 *  uFDMVHM[nodes1+nodes2+2]  - 18 * uFDMVHM[nodes1+nodes2+1] +  11 * uFDMVHM[nodes1+nodes2] ) / 6
    print(upfd)
    print(uppd)
    print(abs(uppd)-abs(upfd))
    print("FDM")
    upfdl = (-2 * uFD[nodes1+nodes2-4]   + 9 *  uFD[nodes1+nodes2-3]  - 18 * uFD[nodes1+nodes2-2] +  11 * uFD[nodes1+nodes2-1] ) / 6
    upfdr = (-2 * uFD[nodes1+nodes2+2]   + 9 *  uFD[nodes1+nodes2+1]  - 18 * uFD[nodes1+nodes2] +  11 * uFD[nodes1+nodes2-1] ) / 6
    print(upfdl)
    print(upfdr)
    print(abs(upfdl)-abs(upfdr))

    print("-------")

    plt.axvline(x=0.5,c="#536872")
    plt.axvline(x=2.25,c="#536872")

    if example == "Steep":
       
        plt.plot(xFull,uSlice,label=r"$\delta$=1/"+str(int(n/2))+"",c="black",marker=markers[i-start],markevery=n)
        if i == start+3:
            plt.plot(xFull,exactSolution(xFull),label="Exact solution",c="black")
    
    if example == "Quartic" or "Linear-cubic" or example == "Sin" or example == "Cos" or example == "Tan":
        
        if solution == "FDM" :
            uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))

            plt.plot(xFull,uSlice-uFD,label=r"$\delta$=1/"+str(int(n/2))+"",c="black",marker=markers[i-start],markevery=n)
            plt.ylabel("Error in displacement w.r.t. FDM")
            

        elif solution == "Exact":
            uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))
            plt.plot(xFull,uFD-exactSolution(xFull),label=r"LLEM-VHM ($\delta$=1/"+str(int(n/2))+")",c="black",marker=markers[i-start],markevery=n)
            plt.plot(xFull,uSlice-exactSolution(xFull),label=r"LLEM-VHM ($\delta$=1/"+str(int(n/2))+")",c="blue",marker=markers[i-start],markevery=n)
            plt.ylabel("Error in displacement w.r.t. exact solution")

    elif i == 4:
        
        plt.plot(xFull,exactSolution(xFull),label="Exact solution",c="black")
        plt.plot(xFull,uSlice,label=r"LLEM-VHM ($\delta$=1/"+str(int(n/2))+")",c="black",marker=markers[i-4],markevery=n)
        plt.ylabel("Displacement")
        np.savetxt("coupling-"+example.lower()+"-vhm.csv",uSlice)  


   

plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.5f'))
plt.title("Example with "+example.lower()+" solution for VHCM with $m=2$")
plt.legend()
plt.grid()
plt.xlabel("$x$")

if solution == "FDM" :
    if example == "Steep":
         plt.savefig("coupling-"+example.lower()+"-"+str(start)+"-"+str(eps)+"-vhm-moving.pdf",bbox_inches='tight')
    else:
        plt.savefig("coupling-"+example.lower()+"-"+str(start)+"-vhm-moving.pdf",bbox_inches='tight')
else:    
   
    plt.savefig("coupling-"+example.lower()+"-"+str(start)+"-vhm-exact-moving.pdf",bbox_inches='tight')
