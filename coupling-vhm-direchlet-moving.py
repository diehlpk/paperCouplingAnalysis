#!/usr/bin/env python3
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

condition = True


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
        return -( 2/np.sqrt(3)) * ( -6 + 4*x )
    elif example == "Quartic":
        return  -32/9 + 64/9 * x - 64/27 * x * x
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
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(x[i])
    
    force[n-1] = 0
    
    return force

#############################################################################
# Exact solution 
#############################################################################

def exactSolution(x):
    
    if example == "Cubic":
        return (2/3/np.sqrt(3)) * ( 9*x - 9*x*x + 2 * x * x * x )
    elif example == "Quartic":
        return 16/9 * x * x - 32/27 * x * x * x + 16/81 * x * x * x * x
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

    
    M[n-1][n-1] = 1

    M *= 1./(2.*h*h)

    if condition :
        print(np.linalg.cond(M))
        with open("con-fdm-direchlet.txt", "a") as f:
            f.write(str(np.linalg.cond(M))+"\n")
            f.close()

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

    M[n][n-1] = 11 / 6 / h
    M[n][n-2] = -18 / 6 / h
    M[n][n-3] = 9 / 6 / h
    M[n][n-4] = -2 / 6 / h

    M[n][n] = 11 / 6 / h 
    M[n][n+1] = -18 / 6 / h
    M[n][n+2] = 9 / 6 / h
    M[n][n+3] = -2 / 6 / h

    for i in range(n+1,2*n-1):
        M[i][i-1] = -2
        M[i][i] = 4
        M[i][i+1] = -2

    M[2*n-1][2*n-1] = -1 
    M[2*n-1][2*n] = 1

    M[2*n][2*n-1] = 11 / 6 / h
    M[2*n][2*n-2] = -18 / 6 / h
    M[2*n][2*n-3] = 9 / 6 / h
    M[2*n][2*n-4] = -2 / 6 / h

    M[2*n][2*n] = 11 / 6 / h
    M[2*n][2*n+1] = -18 / 6 / h
    M[2*n][2*n+2] = 9 / 6 / h
    M[2*n][2*n+3] = -2 / 6 / h

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

    if condition :
        print(np.linalg.cond(M))
        with open("con-vhm-direchlet.txt", "a") as f:
            f.write(str(np.linalg.cond(M))+"\n")
            f.close()

    return M

markers = ['s','o','x','.']
level = [8,16,32,64]

start = 8
for i in range(start,start+3):
    n = np.power(2,i)
    h = 1./n
    nodes1 = int(0.75/h)+1
    nodes2 = int(1.25/h)+1
    nodes3 = n + 1
    nodesFull = 3 * n + 1

    print(nodesFull,h)
    x1 = np.linspace(0,0.75,nodes1)
    x2 = np.linspace(0.75,2.,nodes2)
    x3 = np.linspace(2,3.,nodes3)
    x = np.array(np.concatenate((x1,x2,x3)))

    xFull = np.linspace(0,3.,nodesFull)

    forceCoupled = forceCoupling(nodes1+nodes2+nodes3,x)
    forceCoupled[nodes1-1] = 0
    forceCoupled[nodes1] = 0

    forceCoupled[nodes1+nodes2-1] = 0
    forceCoupled[nodes1+nodes2] = 0

    uFDMVHM = solve(CouplingFDVHM(nodes1,nodes2,nodes3,h),forceCoupled)
    uSlice = np.array(np.concatenate((uFDMVHM[0:nodes1],uFDMVHM[nodes1+1:nodes1+nodes2],uFDMVHM[nodes1+nodes2+1:nodes1+nodes2+nodes3])))

    plt.axvline(x=0.75,c="#536872")
    plt.axvline(x=2,c="#536872")
    
    if example == "Quartic" :

        uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))

        plt.plot(xFull,uSlice-uFD,label=r"$\delta=1/"+str(int(n/2))+"$",c="black",marker=markers[i-start],markevery=level[i-start])
        plt.ylabel("Error in displacement w.r.t. FDM")
        
    
    elif i == 4:

        plt.plot(xFull,exactSolution(xFull),label="Exact solution",c="black")
        plt.plot(xFull,uSlice,label=r"LLEM-VHM ($\delta$=1/"+str(int(n/2))+")",c="black",marker=markers[i-start],markevery=level[i-start])
        plt.ylabel("Displacement")
        np.savetxt("coupling-"+example.lower()+"-vhm-direchlet.csv",uSlice)     

plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.6f'))
plt.title("Example with "+example.lower()+" solution for VHCM with $m=2$")
plt.legend()
plt.grid()
plt.xlabel("$x$")

plt.savefig("coupling-"+example.lower()+"-vhm-direchlet-moving.pdf",bbox_inches='tight')

