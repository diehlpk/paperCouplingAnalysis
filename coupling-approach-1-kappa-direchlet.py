#!/usr/bin/env python3
# Coupling using the displacement for MDCM
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
        print("Error: Either provide Linear, Quadratic, Quartic, or Cubic")
        sys.exit()

def forceFull(n,h):
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(i * h)
    
    force[n-1] = 0
    
    return force

def forceCoupling(n,x):
    
    force = np.zeros(3*n+4)
   
    for i in range(1,3*n+4):
        force[i] = f(x[i])
    
    force[3*n+3] = 0
    
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

    M[3*n-1][3*n-1] = 1

    M *= 1./(2.*h*h)
    
    return M

#############################################################################
# Assemble the stiffness matrix for the coupling of FDM - Displacement - FDM 
#############################################################################

def Coupling(n,h,PDcoef):

    M = np.zeros([3*n+4,3*n+4])

    fFD =  1./(2.*h*h)
    # Boundary

    M[0][0] = 1

    # FD 

    for i in range(1,n-1):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Overlapp

    M[n-1][n-1] = -1 
    M[n-1][n+2] = 1 

    M[n][n] = -1 
    M[n][n-3] = 1 

    M[n+1][n+1] = -1 
    M[n+1][n-2] = 1 

    # PD

    for i in range(n+2,2*n+2):
        M[i][i-2] = 1.  * PDcoef
        M[i][i-1] = 4. * PDcoef
        M[i][i] = -10. * PDcoef
        M[i][i+1] =  4. * PDcoef
        M[i][i+2] = 1. * PDcoef

    # Overlap

    M[2*n+2][2*n+2] = -1
    M[2*n+2][2*n+5] = 1

    M[2*n+3][2*n+3] = -1
    M[2*n+3][2*n+6] = 1

    M[2*n+4][2*n+4] = -1
    M[2*n+4][2*n+1] = 1

    # FD

    for i in range(2*n+5,3*n+3):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Boundary
 
    M[3*n+3][3*n+3] = 1

    return M


markers = ['s','o','x','.']

for i in range(1,5):


    n = np.power(2,4)
    h = 1./n
    nodes = n + 1
    nodesFull = 3 * n + 1

    delta = 2*h

    print(delta)

    #Compute kappa
    E=1
    xj=0.8+(i-1)*0.2
    print(xj)
    kappa=2*E/delta/delta/(1+16*delta*delta/3/(96-192*xj+64*xj*xj))
    PDcoef=-kappa*delta*delta/2/8/h/h

    print(kappa,PDcoef)
    x1 = np.linspace(0,1,nodes)
    x2 = np.linspace(1-2*h,2+2*h,nodes+4)
    x3 = np.linspace(2,3.,nodes)
    x = np.array(np.concatenate((x1,x2,x3)))

    xFull = np.linspace(0,3.,nodesFull)

  
    forceCoupled = forceCoupling(nodes,x)

    forceCoupled[nodes-1] = 0
    forceCoupled[nodes] = 0
    forceCoupled[nodes+1] = 0

    forceCoupled[2*nodes+2] = 0
    forceCoupled[2*nodes+3] = 0
    forceCoupled[2*nodes+4] = 0

    uFDMVHM = solve(Coupling(nodes,h,PDcoef),forceCoupled)
    uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))

    uSlice = np.array(np.concatenate((uFDMVHM[0:nodes],uFDMVHM[nodes+3:2*nodes+2],uFDMVHM[2*nodes+5:len(x)])))

    plt.axvline(x=1,c="#536872")
    plt.axvline(x=2,c="#536872")

    if example == "Quartic" or "Linear-cubic":

        plt.plot(xFull,uSlice-uFD,label=r"$\kappa$="+str(np.round(kappa,3))+"",c="black",marker=markers[i-1],markevery=n)
        plt.ylabel("Error in displacement w.r.t. FDM")

    elif i == 4:

        plt.plot(xFull,uFD,label="FDM",c="black")
        plt.plot(xFull,uSlice,label=r"$\delta$=1/"+str(int(n/2))+"",c="black",marker=markers[i-1],markevery=n)
        plt.ylabel("Displacement")
        np.savetxt("coupling-"+example.lower()+"-approach-1.csv",uSlice)   

n = np.power(2,4)
h = 1./n
nodes = n + 1
nodesFull = 3 * n + 1
kappa = 128
PDcoef=-kappa*delta*delta/2/8/h/h
print(kappa,PDcoef)

x1 = np.linspace(0,1,nodes)
x2 = np.linspace(1-2*h,2+2*h,nodes+4)
x3 = np.linspace(2,3.,nodes)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull)

  
forceCoupled = forceCoupling(nodes,x)

forceCoupled[nodes-1] = 0
forceCoupled[nodes] = 0
forceCoupled[nodes+1] = 0

forceCoupled[2*nodes+2] = 0
forceCoupled[2*nodes+3] = 0
forceCoupled[2*nodes+4] = 0

uFDMVHM = solve(Coupling(nodes,h,PDcoef),forceCoupled)
uFD = solve(FDM(nodesFull,h),forceFull(nodesFull,h))

uSlice = np.array(np.concatenate((uFDMVHM[0:nodes],uFDMVHM[nodes+3:2*nodes+2],uFDMVHM[2*nodes+5:len(x)])))

plt.plot(xFull,uSlice-uFD,label=r"$\kappa$="+str(kappa)+"",c="black")


plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.5f')) 
plt.title("Influence of $\kappa$ for MDCM with $\delta=1/"+str(n)+"$")
plt.legend()
plt.grid()
plt.xlabel("$x$")


plt.savefig("coupling-"+example.lower()+"-approach-1-kappa-direchlet.pdf",bbox_inches='tight')

