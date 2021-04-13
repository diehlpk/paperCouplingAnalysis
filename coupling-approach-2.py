#!/usr/bin/env python3
# Coupling using the stress's first order approximation  (MSCM)
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 03/02/2021
import numpy as np
import sys 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

pgf_with_latex = {"text.usetex": True, "font.size" : 12, "pgf.preamble" : [r'\usepackage{xfrac}'] }


example = sys.argv[1]

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
    elif example == "Linear-cubic":
        g = 31./4.
        if x < 1.5:
            return 0 
        else:
            return 9-6*x
    elif example == "Linear-quartic" :
        g = 29./2.
        if x < 1.5:
            return 0
        else:
            return -12 * (x-1.5)  * (x-1.5)
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
    
    force = np.zeros(3*n+4)
   
    for i in range(1,3*n+4):
        force[i] = f(x[i])
    

    force[3*n+3] = g
    
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
    elif example == "Linear-quartic":
        return 0
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

    M[3*n-1][3*n-1] = 3*h
    M[3*n-1][3*n-2] = -4*h
    M[3*n-1][3*n-3] = h

    M *= 1./(2.*h*h)
 
    return M

#############################################################################
# Assemble the stiffness matrix for the coupling of FDM - Displacement - FDM 
#############################################################################

def Coupling(n,h):

    M = np.zeros([3*n+4,3*n+4])

    fFD =  1./(2.*h*h)
    fPD =  1./(8.*h*h)

    M[0][0] = 1

    # FD 

    for i in range(1,n-1):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Overlapp

    # 1
    M[n-1][n-1] = 1
    M[n-1][n+2] =  -1

    # 0.5
    M[n][n] = 11 / 6 / h
    M[n][n+1] = -18 / 6 / h
    M[n][n+2] = 9 / 6 / h
    M[n][n+3] = -2 / 6 / h

    M[n][n-6] =  -2 / 6 / h
    M[n][n-5] =  9 / 6 / h
    M[n][n-4] = -18  / 6 / h
    M[n][n-3] = 11 / 6 / h

    # 0.75
    M[n+1][n+1] = 11 / 6 / h
    M[n+1][n+2] = - 18 / 6 / h
    M[n+1][n+3] = 9 / 6 / h
    M[n+1][n+4] = -2 / 6 / h


    M[n+1][n-2] = 11 / 6 / h
    M[n+1][n-3] = -18 / 6 / h
    M[n+1][n-4] = 9 / 6 / h
    M[n+1][n-5] = -2 / 6 / h

    # 1
    M[n+2][n+2] = 11 / 6 / h
    M[n+2][n+3] = -18 / 6 / h
    M[n+2][n+4] = 9 / 6 / h
    M[n+2][n+5] = -2 / 6 / h

    M[n+2][n-1] = 11 / 6 / h
    M[n+2][n-2] = -18 / 6 / h
    M[n+2][n-3] = 9 / 6 / h
    M[n+2][n-4] = -2 / 6 / h

    # PD

    for i in range(n+3,2*n+1):
        M[i][i-2] = -1.  * fPD
        M[i][i-1] = -4. * fPD
        M[i][i] = 10. * fPD
        M[i][i+1] =  -4. * fPD
        M[i][i+2] = -1. * fPD

    # Overlap

    # 2
    M[2*n+1][2*n+1] = -1
    M[2*n+1][2*n+4] = 1

    # 2.25
    M[2*n+2][2*n+2] = -11 / 6 / h
    M[2*n+2][2*n+1] = 18 / 6 / h
    M[2*n+2][2*n] = -9 / 6 / h
    M[2*n+2][2*n-1] = 2 / 6 / h


    M[2*n+2][2*n+8] =  2 / 6 / h 
    M[2*n+2][2*n+7] =  -9 / 6 / h 
    M[2*n+2][2*n+6] = 18  / 6 / h
    M[2*n+2][2*n+5] = -11  / 6 / h

    # 2.5

    M[2*n+3][2*n+3] = -11 / 2 / h
    M[2*n+3][2*n+2] =  18 / 2 / h
    M[2*n+3][2*n+1] = -9 / 2 / h
    M[2*n+3][2*n+1] = -9 / 2 / h

    M[2*n+3][2*n+6] = -11 / 6 / h
    M[2*n+3][2*n+7] = 18 / 6 / h
    M[2*n+3][2*n+8] = -9 / 6 / h
    M[2*n+3][2*n+9] = 2 / 6 / h

    # 2

    M[2*n+4][2*n+1] = -11 / 6 / h
    M[2*n+4][2*n] = 18 / 6 / h
    M[2*n+4][2*n-1] = -9 / 6 / h
    M[2*n+4][2*n-2] = 2 / 6 / h

    M[2*n+4][2*n+4] = -11 / 6 / h
    M[2*n+4][2*n+5] = 18  / 6 / h
    M[2*n+4][2*n+6] = -9 / 6 / h
    M[2*n+4][2*n+7] = 2 / 6 / h


    # FD

    for i in range(2*n+5,3*n+3):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Boundary

    M[3*n+3][3*n+3] = 11 / 6 / h
    M[3*n+3][3*n+2] = -18 / 6 / h
    M[3*n+3][3*n+1] = 9 / 6 / h
    M[3*n+3][3*n]  = -2 / 6 / h


    np.savetxt("pd2.csv", M, delimiter=",")

    return M


markers = ['s','o','x','.']

for i in range(4,8):
    n = np.power(2,i)
    h = 1./n
    nodes = n + 1
    nodesFull = 3 * n + 1

    print(nodes,h)
    x1 = np.linspace(0,1,nodes)
    x2 = np.linspace(1-2*h,2+2*h,nodes+4)
    x3 = np.linspace(2,3.,nodes)
    x = np.array(np.concatenate((x1,x2,x3)))

    xFull = np.linspace(0,3.,nodesFull)

  
    forceCoupled = forceCoupling(nodes,x)

    forceCoupled[nodes-1] = 0
    forceCoupled[nodes] = 0
    forceCoupled[nodes+1] = 0
    forceCoupled[nodes+2] = 0

    forceCoupled[2*nodes+1] = 0
    forceCoupled[2*nodes+2] = 0
    forceCoupled[2*nodes+3] = 0
    forceCoupled[2*nodes+4] = 0


    uFDMVHM = solve(Coupling(nodes,h),forceCoupled)
    

    uSlice = np.array(np.concatenate((uFDMVHM[0:nodes],uFDMVHM[nodes+3:2*nodes+2],uFDMVHM[2*nodes+5:len(x)])))

    if example == "Linear" or example == "Quadratic" or example == "Cubic":

        if i == 4 :

            plt.plot(xFull,exactSolution(xFull),label="Exact",c="black")
            plt.plot(xFull,uSlice,label=r"LLEM-PDM ($\delta$=1/"+str(int(n/2))+")",c="black",marker=markers[i-4],markevery=n)
            plt.ylabel("Displacement")
            np.savetxt("coupling-"+example.lower()+"-approach-2.csv",uSlice)   
   
    else:

        uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))

        plt.plot(xFull,uSlice-uFD,label=r"LLEM-PDM ($\delta$=1/"+str(int(n/2))+")",c="black",marker=markers[i-4],markevery=n)
        plt.ylabel("Error in displacement w.r.t. FDM")
        

plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.5f'))    
plt.title("Example with "+example.lower()+" solution for MSCM with $m=2$")
plt.legend()
plt.grid()
plt.xlabel("$x$")

plt.savefig("coupling-"+example.lower()+"-approach-2-1.pdf",bbox_inches='tight')

