#!/usr/bin/env python3
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 02/05/2021
import numpy as np
import sys 
import matplotlib.pyplot as plt
import matplotlib

pgf_with_latex = {"text.usetex": True, "font.size" : 12, "pgf.preamble" : [r'\usepackage{xfrac}'] }


example = sys.argv[1]


#############################################################################
# Solve the system
#############################################################################

def solve(M,f):
    return np.linalg.solve(M,f)

#############################################################################
# Loading
#############################################################################

def f(x):
    
    if example == "Cubic":
        return -( np.power(2,np.sqrt(3))) * ( -6 + 4*x )
    elif example == "Quartic":
        return  - 32 + 64 * x - 64/3 * x * x
    elif example == "Quadratic":
        return 8/9
    else:
        print("Error: Either provide Quadratic, Quartic, or Cubic")
        sys.exit()

def forceFull(n,h):
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(i * h)
    
    force[n-1] = 0
    
    return force

def forceCoupling(n,x):
    
    force = np.zeros(3*n)
   
    for i in range(1,3*n-1):
        force[i] = f(x[i])
    
    force[3*n-1] = 0
    
    return force

#############################################################################
# Exact solution 
#############################################################################

def exactSolution(x):
    
    if example == "Cubic":
        return (2/3*np.sqrt(3)) * ( 9*x - 9*x*x + 2 * x * x * x )
    elif example == "Quartic":
        return 16 * x*x - 32/3 * x*x*x + 16/9 * x * x * x * x 
    elif example == "Quadratic":
        return  4/3 * x - 4/9 * x * x
    else:
        print("Error: Either provide Quadratic, Quartic, or Cubic")
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

    M[n-1][n-1] = 3*h
    M[n-1][n-2] = -4*h
    M[n-1][n-3] = h

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

    M[3*n-1][3*n-1] = 3*h
    M[3*n-1][3*n-2] = -4*h
    M[3*n-1][3*n-3] = h

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

    M[n][n-1] = 3*h * fFDM
    M[n][n-2] = -4*h * fFDM
    M[n][n-3] = 1*h * fFDM

    M[n][n] = 12*h  * fVHM
    M[n][n+1] = -16*h  * fVHM
    M[n][n+2] = 4*h  * fVHM

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

    M[2*n][2*n-1] = 12*h * fVHM
    M[2*n][2*n-2] = -16*h * fVHM
    M[2*n][2*n-3] = 4*h * fVHM

    M[2*n][2*n] = 3*h * fFDM
    M[2*n][2*n+1] = -4*h * fFDM
    M[2*n][2*n+2] = 1*h * fFDM

    for i in range(2*n+1,3*n-1):
        M[i][i-1] = -2 * fFDM
        M[i][i] = 4 * fFDM
        M[i][i+1] = -2 * fFDM

    M[3*n-1][3*n-1] = 1
    #M[3*n-1][3*n-1] = 3*h * fFDM
    #M[3*n-1][3*n-2] = -4*h * fFDM
    #M[3*n-1][3*n-3] = h * fFDM
    
    return M

markers = ['s','o','x','.']

for i in range(3,7):
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
    
    
    if example == "Quartic" or example == "Cubic":

        uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))

        plt.plot(xFull,uSlice-exactSolution(xFull),label=r"LLEM-VHM ($\delta$="+str(2*h)+")",c="black",marker=markers[i-3],markevery=5)
        plt.ylabel("Error in displacement w.r.t exact solution")

        
    
    elif i == 3:

        plt.plot(xFull,exactSolution(xFull),label="Exact solution",c="black")
        plt.plot(xFull,uSlice,label=r"LLEM-VHM ($\delta$="+str(2*h)+")",c="black",marker=markers[i-3],markevery=5)
        plt.ylabel("Displacement")

plt.title("Example with "+example+" solution for Problem (19)")
plt.legend()
plt.grid()
plt.xlabel("$x$")

plt.savefig("coupling-"+example.lower()+"-vhm-direchlet.pdf",bbox_inches='tight')

