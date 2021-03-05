#!/usr/bin/env python3
# Coupling using the stress's first order approximation  (Problem (18))
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 03/02/2021
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
        return x
    elif example == "Quartic":
        return x*x
    elif example == "Quadratic":
        return 1
    elif example == "Linear":
        return 0
    else:
        print("Error: Either provide Linear, Quadratic, Quartic, or Cubic")
        sys.exit()

def forceFull(n,h):
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(i * h)
    
    force[n-1] = 1
    
    return force

def forceCoupling(n,x):
    
    force = np.zeros(3*n+4)
   
    for i in range(1,3*n+4):
        force[i] = f(x[i])
    
    force[3*n+3] = 1
    
    return force

#############################################################################
# Exact solution 
#############################################################################

def exactSolution(x):
    
    if example == "Cubic":
        return - x *x * x / 6 + 5.5 * x
    elif example == "Quartic":
        return - x * x * x * x / 12 + 10 * x
    elif example == "Quadratic":
        return -0.5 * x * x + 4 *x
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

    #np.savetxt("fd.csv", M, delimiter=",")
    
    return M

#############################################################################
# Assemble the stiffness matrix for the coupling of FDM - Displacement - FDM 
#############################################################################

def Coupling(n,h):

    M = np.zeros([3*n+4,3*n+4])

    fFD =  1./(2.*h*h)
    fPD =  1./(8.*h*h)

    fFD = 1
    fPD = 1
    h = 1
    # Boundary

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
    M[n][n+1] = h
    M[n][n] = -h 
    M[n][n-1] =  1 * h
    M[n][n-2] = -4  * h
    M[n][n-3] = 3 * h 

    # 0.75
    M[n+1][n+1] = - h
    M[n+1][n+2] = h  
    M[n+1][n-2] = 3 * h
    M[n+1][n-1] = -4 * h
    M[n+1][n-0] = 1 * h

    # 1
    M[n+2][n+2] = -h
    M[n+2][n+3] = h
    M[n+2][n-1] = 3 * h
    M[n+2][n] = -4 *h
    M[n+2][n+1] = 1 *h

    # PD

    for i in range(n+3,2*n+1):
        M[i][i-2] = -1.  * fPD
        M[i][i-1] = -4. * fPD
        M[i][i] = 10. * fPD
        M[i][i+1] =  -4. * fPD
        M[i][i+2] = -1. * fPD

    # Overlap

    # 2
    M[2*n+1][2*n+1] = 1
    M[2*n+1][2*n+4] =  -1

    # 0.5
    M[2*n+2][2*n+3] = h
    M[2*n+2][2*n+2] = -h 
    M[2*n+2][2*n-1] =  1 * h
    M[2*n+2][2*n-2] = -4  * h
    M[2*n+2][2*n-3] = 3 * h 

    # 0.75
    M[2*n+3][2*n+1] = - h
    M[2*n+3][2*n+2] = h  
    M[2*n+3][2*n-2] = 3 * h
    M[2*n+3][2*n-1] = -4 * h
    M[2*n+3][2*n-0] = 1 * h

    # 1
    M[2*n+4][2*n+2] = -h
    M[2*n+4][2*n+3] = h
    M[2*n+4][2*n-1] = 3 * h
    M[2*n+4][2*n] = -4 *h
    M[2*n+4][2*n+1] = 1 *h

    #M[2*n+2][2*n+2] = -1
    #M[2*n+2][2*n+5] = 1

    #M[2*n+3][2*n+3] = -1
    #M[2*n+3][2*n+6] = 1

    #M[2*n+4][2*n+4] = -1
    #M[2*n+4][2*n+1] = 1

    # FD

    for i in range(2*n+5,3*n+3):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Boundary

    M[3*n+3][3*n+3] = 3*h * fFD
    M[3*n+3][3*n+2] = -4*h * fFD
    M[3*n+3][3*n+1] = h * fFD

    np.savetxt("pd2.csv", M, delimiter=",")

    return M


markers = ['s','o','x','.']

for i in range(2,3):
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
    #forceCoupled[nodes+2] = 0


    forceCoupled[2*nodes+2] = 0
    forceCoupled[2*nodes+3] = 0
    forceCoupled[2*nodes+4] = 0
    #forceCoupled[2*nodes+5] = 0

    uFDMVHM = solve(Coupling(nodes,h),forceCoupled)

    uSlice = np.array(np.concatenate((uFDMVHM[0:nodes],uFDMVHM[nodes+3:2*nodes+2],uFDMVHM[2*nodes+5:len(x)])))

    plt.scatter(xFull,uSlice,label=r"LLEM-PDM ($\delta$="+str(2*h)+")",c="black",marker=markers[i-2])

    #if i == 5:

        #xFull = np.linspace(0,3.,nodesFull)
        #print(xFull)

        #uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))
        #uVHM = solve(VHM(nodesFull,h),forceFull(nodesFull,h))
        #uFDFD = solve(CouplingFDFD(nodes,h),forceCoupled)
        #plt.plot(xFull,uFD,label="FDM")
        #plt.plot(xFull,uVHM,label="VHM")
        #plt.plot(xFull,uFD,label="FDM-FDM")
    plt.plot(xFull,exactSolution(xFull),label="Exact")


    
plt.title("Example with "+example+" solution for Problem (18) using $\sigma_1$")
plt.legend()
plt.grid()
plt.xlabel("$x$")
plt.ylabel("Error in displacement")

plt.savefig("coupling-"+example.lower()+"-approach-2-1.pdf",bbox_inches='tight')
