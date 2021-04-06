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
    else:
        print("Error: Either provide Linear, Quadratic, Quartic, or Cubic")
        sys.exit()

def forceFull(n,h):
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(i * h)
    
    force[n-1] = g
    
    return force

def forceCoupling(n,x):
    
    force = np.zeros(3*n)
   
    for i in range(1,3*n-1):
        force[i] = f(x[i])
    
    force[3*n-1] = g
    
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

    M[n-1][n-1] = 11*h / 3 
    M[n-1][n-2] = -18*h / 3
    M[n-1][n-3] = 9* h / 3
    M[n-1][n-4] = -2* h / 3


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

def CouplingFDVHM(n,h):

    fVHM = 1./(8.*h*h)
    fFDM = 1./(2.*h*h)

    M = np.zeros([3*n,3*n])
    
    M[0][0] = 1 * fFDM

    for i in range(1,n-1):
        M[i][i-1] = -2 * fFDM
        M[i][i] = 4 * fFDM
        M[i][i+1] = -2 * fFDM

    M[n-1][n-1] = -1 
    M[n-1][n] = 1   

    M[n][n-1] = 11*h * fFDM / 3
    M[n][n-2] = -18*h * fFDM / 3
    M[n][n-3] = 9*h * fFDM / 3
    M[n][n-4] = -2*h * fFDM / 3

    M[n][n] = 11 / 6 / h 
    M[n][n+1] = -18 / 6 / h
    M[n][n+2] = 9 / 6 / h
    M[n][n+3] = -2 / 6 / h


    M[n+1][n] = -8 * fVHM
    M[n+1][n+1] = 16 * fVHM
    M[n+1][n+2] = -8 * fVHM

    for i in range(n+2,2*n-2):
        M[i][i-2] = -1. * fVHM
        M[i][i-1] = -4. * fVHM
        M[i][i] = 10. * fVHM
        M[i][i+1] =  -4. * fVHM
        M[i][i+2] = -1. * fVHM

    M[2*n-2][2*n-3] = -8 * fVHM
    M[2*n-2][2*n-2] = 16 * fVHM
    M[2*n-2][2*n-1] = -8 * fVHM

    M[2*n-1][2*n-1] = -1 
    M[2*n-1][2*n] = 1  

    M[2*n][2*n-1] = 11 / 6 / h
    M[2*n][2*n-2] = -18 / 6 / h
    M[2*n][2*n-3] = 9 / 6 / h
    M[2*n][2*n-4] = -2 / 6 / h

    M[2*n][2*n] = 11*h * fFDM / 3
    M[2*n][2*n+1] = -18*h * fFDM / 3
    M[2*n][2*n+2] = 9*h * fFDM / 3
    M[2*n][2*n+3] = -2*h * fFDM / 3

    for i in range(2*n+1,3*n-1):
        M[i][i-1] = -2 * fFDM
        M[i][i] = 4 * fFDM
        M[i][i+1] = -2 * fFDM

    M[3*n-1][3*n-1] = 11*h * fFDM / 3
    M[3*n-1][3*n-2] = -18*h * fFDM / 3 
    M[3*n-1][3*n-3] = 9 * h * fFDM / 3
    M[3*n-1][3*n-4] = -2 * h * fFDM / 3
    
    return M

markers = ['s','o','x','.']

for i in range(4,8):
    n = np.power(2,i)
    h = 1./n
    nodes = n + 1
    nodesFull = 3 * n + 1

    print(nodes,h)
    x1 = np.linspace(0,1,nodes)
    x2 = np.linspace(1,2.,nodes)
    x3 = np.linspace(2,3.,nodes)
    x = np.array(np.concatenate((x1,x2,x3)))

    xFull = np.linspace(0,3.,nodesFull)

    forceCoupled = forceCoupling(nodes,x)
    forceCoupled[n] = 0
    forceCoupled[n+1] = 0

    forceCoupled[2*n+1] = 0
    forceCoupled[2*n+2] = 0


    uFDMVHM = solve(CouplingFDVHM(nodes,h),forceCoupled)
    uSlice = np.array(np.concatenate((uFDMVHM[0:nodes],uFDMVHM[nodes+1:2*nodes],uFDMVHM[2*nodes+1:3*nodes])))
    
    
    if example == "Quartic":

        if solution == "FDM" :

            uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))

            plt.plot(xFull,uSlice-uFD,label=r"LLEM-VHM ($\delta$=1/"+str(int(n/2))+")",c="black",marker=markers[i-4],markevery=n)
            plt.ylabel("Error in displacement w.r.t. FDM")

        elif solution == "Exact":

            plt.plot(xFull,uSlice-exactSolution(xFull),label=r"LLEM-VHM ($\delta$=1/"+str(int(n/2))+")",c="black",marker=markers[i-4],markevery=n)
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
    plt.savefig("coupling-"+example.lower()+"-vhm.pdf",bbox_inches='tight')
else:
    plt.savefig("coupling-"+example.lower()+"-vhm-exact.pdf",bbox_inches='tight')
