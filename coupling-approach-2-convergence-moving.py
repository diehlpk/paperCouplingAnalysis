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
case = sys.argv[2]
factor = sys.argv[3]

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
    else:
        print("Error: Either provide Linear, Quadratic, Quartic, or Cubic")
        sys.exit()

def forceFull(n,h):
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(i * h)
    
    force[n-1] = g
    
    return force

def forceCoupling(n,x,m):
    
    force = np.zeros(n+2*m)
   
    for i in range(1,n+2*m):
        force[i] = f(x[i])
    
    force[n+2*m-1] = g
    
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

def Coupling(nodes1,nodes2,nodes3,h):

    total = nodes1 + nodes2 + nodes3

    M = np.zeros([total+4,total+4])

    fFD =  1./(2.*h*h)
    fPD =  1./(8.*h*h)

    M[0][0] = 1

    # FD 
    n = nodes1

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

    for i in range(n+3,nodes1+nodes2+1):
        M[i][i-2] = -1.  * fPD
        M[i][i-1] = -4. * fPD
        M[i][i] = 10. * fPD
        M[i][i+1] =  -4. * fPD
        M[i][i+2] = -1. * fPD

    # Overlap

    n += nodes2

    # 2
    M[n+1][n+1] = -1
    M[n+1][n+4] = 1

    # 2.25
    M[n+2][n+2] = -11 / 6 / h
    M[n+2][n+1] = 18 / 6 / h
    M[n+2][n] = -9 / 6 / h
    M[n+2][n-1] = 2 / 6 / h

    M[n+2][n+8] =  2 / 6 / h 
    M[n+2][n+7] =  -9 / 6 / h 
    M[n+2][n+6] = 18  / 6 / h
    M[n+2][n+5] = -11  / 6 / h

    # 2.5

    M[n+3][n+3] = -11 / 2 / h
    M[n+3][n+2] =  18 / 2 / h
    M[n+3][n+1] = -9 / 2 / h
    M[n+3][n+1] = -9 / 2 / h

    M[n+3][n+6] = -11 / 6 / h
    M[n+3][n+7] = 18 / 6 / h
    M[n+3][n+8] = -9 / 6 / h
    M[n+3][n+9] = 2 / 6 / h

    # 2

    M[n+4][n+1] = -11 / 6 / h
    M[n+4][n] = 18 / 6 / h
    M[n+4][n-1] = -9 / 6 / h
    M[n+4][n-2] = 2 / 6 / h

    M[n+4][n+4] = -11 / 6 / h
    M[n+4][n+5] = 18  / 6 / h
    M[n+4][n+6] = -9 / 6 / h
    M[n+4][n+7] = 2 / 6 / h

    # FD

    for i in range(n+5,n+nodes3+3):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Boundary
    n += nodes3

    # Boundary

    M[n+3][n+3] = 11 / 6 / h
    M[n+3][n+2] = -18 / 6 / h
    M[n+3][n+1] = 9 / 6 / h
    M[n+3][n]  = -2 / 6 / h

    return M


def Coupling4(nodes1,nodes2,nodes3,h):

    total = nodes1 + nodes2 + nodes3

    M = np.zeros([total+8,total+8])

    fFD =  1./(2.*h*h)

    # Boundary

    M[0][0] = 1

    # FD 
    n = nodes1

    for i in range(1,n-1):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Overlapp

    M[n-1][n-1] = -1
    M[n-1][n+4] = 1

    #
    M[n][n] = 11 / 6 / h
    M[n][n+1] = -18 / 6 / h
    M[n][n+2] = 9 / 6 / h
    M[n][n+3] = -2 / 6 / h

    M[n][n-5] = 11 / 6 / h
    M[n][n-6] = -18 / 6 / h
    M[n][n-7] = 9 / 6 / h
    M[n][n-8] = -2 / 6 / h

    #
    M[n+1][n+1] = 11 / 6 / h
    M[n+1][n+2] = -18 / 6 / h
    M[n+1][n+3] = 9 / 6 / h
    M[n+1][n+4] = -2 / 6 / h

    M[n+1][n-4] = 11 / 6 / h
    M[n+1][n-5] = -18 / 6 / h
    M[n+1][n-6] = 9 / 6 / h
    M[n+1][n-7] = -2 / 6 / h

    #
    M[n+2][n+2] = 11 / 6 / h
    M[n+2][n+3] = -18 / 6 / h
    M[n+2][n+4] = 9 / 6 / h
    M[n+2][n+5] = -2 / 6 / h

    M[n+2][n-3] = 11 / 6 / h
    M[n+2][n-4] = -18 / 6 / h
    M[n+2][n-5] = 9 / 6 / h
    M[n+2][n-6] = -2 / 6 / h 

    M[n+3][n+3] =  11 / 6 / h
    M[n+3][n+4] = -18 / 6 / h
    M[n+3][n+5] = 9 / 6 / h
    M[n+3][n+6] = -2 / 6 / h 


    M[n+3][n-2] = 11 / 6 / h
    M[n+3][n-3] = -18 / 6 / h
    M[n+3][n-4] = 9 / 6 / h
    M[n+3][n-5] = -2 / 6 / h 

    # PD

    for i in range(n+4,nodes1+nodes2+4):

        M[i][i-4] = -(1./64.)/ 2. / h / h *2
        M[i][i-3] =  -(1./24.)/ 2. / h / h*2
        M[i][i-2] = -(1./16.)/ 2. / h / h*2
        M[i][i-1] = -(1./8.)/ 2. / h / h*2
        M[i][i] = (47./96.)/ 2. / h / h*2
        M[i][i+1] =  -(1./8.)/ 2. / h / h*2
        M[i][i+2] = -(1./16.)/ 2. / h / h*2
        M[i][i+3] =  -(1./24.)/ 2. / h / h*2
        M[i][i+4] = -(1./64.)/ 2. / h / h *2

    n += nodes2

    # Overlap
    M[n+4][n+4] = -1
    M[n+4][n+9] = 1

    #
    M[n+5][n+2] = 2 / 6 / h
    M[n+5][n+3] = -9 / 6 / h
    M[n+5][n+4] = 18 / 6 / h
    M[n+5][n+5] = -11 / 6 / h

    M[n+5][n+10] = -11 / 6 / h
    M[n+5][n+11] = 18 / 6 / h
    M[n+5][n+12] = -9 / 6 / h
    M[n+5][n+13] = 2 / 6 / h

    #
    M[n+6][n+3] = 2 / 6 / h
    M[n+6][n+4] = -9 / 6 / h
    M[n+6][n+5] = 18 / 6 / h
    M[n+6][n+6] = -11 / 6 / h

    M[n+6][n+11] = -11 / 6 / h
    M[n+6][n+12] = 18 / 6 / h
    M[n+6][n+13] = -9 / 6 / h
    M[n+6][n+14] = 2 / 6 / h

    #
    M[n+7][n+4] = 2 / 6 / h
    M[n+7][n+5] = -9 / 6 / h
    M[n+7][n+6] = 18 / 6 / h
    M[n+7][n+7] = -11 / 6 / h

    M[n+7][n+12] = -11 / 6 / h
    M[n+7][n+13] = 18 / 6 / h
    M[n+7][n+14] = -9 / 6 / h
    M[n+7][n+15] = 2 / 6 / h

    #
    M[n+8][n+11] = 2 / 6 / h
    M[n+8][n+10] = -9 / 6 / h
    M[n+8][n+9] = 18 / 6 / h
    M[n+8][n+8] = -11 / 6 / h
    
    M[n+8][n+3] = -11 / 6 / h
    M[n+8][n+2] = 18 / 6 / h
    M[n+8][n+1] = -9 / 6 / h
    M[n+8][n] = 2 / 6 / h

    
    # FD

    for i in range(n+9,n+nodes3+7):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Boundary
    n += nodes3

    M[n+7][n+7] = 11 *  h * fFD / 3
    M[n+7][n+6] =  -18 * h * fFD  / 3
    M[n+7][n+5] = 9 * h * fFD / 3
    M[n+7][n+4] = -2 * h * fFD / 3

    return M

def Coupling8(nodes1,nodes2,nodes3,h):

    total = nodes1 + nodes2 + nodes3

    M = np.zeros([total+16,total+16])

    fFD =  1./(2.*h*h)

    # Boundary

    M[0][0] = 1

    # FD 

    n = nodes1

    for i in range(1,n-1):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Overlapp

    M[n-1][n-1] = 1
    M[n-1][n+8] = -1

    #
    M[n][n+3] = -2 / 6 / h
    M[n][n+2] = 9 / 6 / h
    M[n][n+1] = -18 / 6 / h
    M[n][n] = 11 / 6 / h

    M[n][n-9] = 11 / 6 / h
    M[n][n-10] =  -18 / 6 / h
    M[n][n-11] = 9 / 6 / h
    M[n][n-12] = -2 / 6 / h

    #
    M[n+1][n+4] = -2 / 6 / h
    M[n+1][n+3] = 9 / 6 / h
    M[n+1][n+2] = -18 / 6 / h
    M[n+1][n+1] = 11 / 6 / h

    M[n+1][n-8] =  11 / 6 / h
    M[n+1][n-9] =  -18 / 6 / h
    M[n+1][n-10] = 9 / 6 / h
    M[n+1][n-11] = -2 / 6 / h

    #
    M[n+2][n+5] = -2 / 6 / h
    M[n+2][n+4] = 9 / 6 / h
    M[n+2][n+3] = -18 / 6 / h
    M[n+2][n+2] = 11 / 6 / h

    M[n+2][n-7] = 11 / 6 / h
    M[n+2][n-8] = -18 / 6 / h
    M[n+2][n-9] = 9 / 6 / h
    M[n+2][n-10] = -2 / 6 / h

    #
    M[n+3][n+6] = -2 / 6 / h
    M[n+3][n+5] = 9 / 6 / h
    M[n+3][n+4] = -18 / 6 / h
    M[n+3][n+3] = 11 / 6 / h

    M[n+3][n-6] = 11 / 6 / h
    M[n+3][n-7] = -18 / 6 / h
    M[n+3][n-8] = 9 / 6 / h
    M[n+3][n-9] = -2 / 6 / h

    #
    M[n+4][n+7] = -2 / 6 / h
    M[n+4][n+6] = 9 / 6 / h
    M[n+4][n+5] = -18 / 6 / h
    M[n+4][n+4] = 11 / 6 / h

    M[n+4][n-5] = 11 / 6 / h
    M[n+4][n-6] = -18 / 6 / h
    M[n+4][n-7] = 9 / 6 / h
    M[n+4][n-8] = -2 / 6 / h

    #
    M[n+5][n+8] = -2 / 6 / h
    M[n+5][n+7] = 9 / 6 / h
    M[n+5][n+6] = -18 / 6 / h
    M[n+5][n+5] = 11 / 6 / h

    M[n+5][n-4] = 11 / 6 / h
    M[n+5][n-5] = -18 / 6 / h
    M[n+5][n-6] = 9 / 6 / h
    M[n+5][n-7] = -2 / 6 / h

    #
    M[n+6][n+9] = -2 / 6 / h
    M[n+6][n+8] = 9 / 6 / h
    M[n+6][n+7] = -18 / 6 / h
    M[n+6][n+6] = 11 / 6 / h

    M[n+6][n-3] = 11 / 6 / h
    M[n+6][n-4] = -18 / 6 / h
    M[n+6][n-5] = 9 / 6 / h
    M[n+6][n-6] =  -2 / 6 / h


    #
    M[n+7][n+10] = -2 / 6 / h
    M[n+7][n+9] = 9 / 6 / h
    M[n+7][n+8] = -18 / 6 / h
    M[n+7][n+7] = 11 / 6 / h

    M[n+7][n-2] = 11 / 6 / h
    M[n+7][n-3] = -18 / 6 / h
    M[n+7][n-4] = 9 / 6 / h
    M[n+7][n-5] =  -2 / 6 / h

    # PD

    for i in range(n+8,n+nodes2+8):
        M[i][i-8] = -(1/512) / 2. / h / h*2
        M[i][i-7] = -(1/224) / 2. / h / h*2
        M[i][i-6] = -(1/192) / 2. / h / h*2
        M[i][i-5] = -(1/160) / 2. / h / h*2
        M[i][i-4] = -(1/128) / 2. / h / h*2
        M[i][i-3] = -(1/96) / 2. / h / h*2
        M[i][i-2] = -(1/64) / 2. / h / h*2
        M[i][i-1] = -(1/32) / 2. / h / h*2
        M[i][i] =  (1487/8960) / 2. / h / h*2
        M[i][i+1] = -(1/32) / 2. / h / h*2
        M[i][i+2] = -(1/64) / 2. / h / h*2
        M[i][i+3] = -(1/96) / 2. / h / h*2
        M[i][i+4] = -(1/128) / 2. / h / h*2
        M[i][i+5] = -(1/160) / 2. / h / h*2
        M[i][i+6] = -(1/192) / 2. / h / h*2
        M[i][i+7] = -(1/224) / 2. / h / h*2
        M[i][i+8] = -(1/512) / 2. / h / h*2

    # Overlap
    n += nodes2

    M[n+8][n+8] = -1
    M[n+8][n+17] = 1

    #
    M[n+9][n+6] = 2 / 6 / h
    M[n+9][n+7] = -9 / 6 / h
    M[n+9][n+8] = 18 / 6 / h
    M[n+9][n+9] = -11 / 6 / h

    M[n+9][n+18] =  -11 / 6 / h
    M[n+9][n+19] = 18 / 6 / h
    M[n+9][n+20] =  -9 / 6 / h
    M[n+9][n+21] = 2 / 6 / h

    #
    M[n+10][n+7] = 2 / 6 / h
    M[n+10][n+8] = -9 / 6 / h
    M[n+10][n+9] =  18 / 6 / h
    M[n+10][n+10] = -11 / 6 / h

    M[n+10][n+19] = -11 / 6 / h
    M[n+10][n+20] = 18 / 6 / h
    M[n+10][n+21] = -9 / 6 / h
    M[n+10][n+22] = 2 / 6 / h

    #
    M[n+11][n+8] = 2 / 6 / h
    M[n+11][n+9] = -9 / 6 / h
    M[n+11][n+10] = 18 / 6 / h
    M[n+11][n+11] = -11 / 6 / h

    M[n+11][n+20] = -11 / 6 / h
    M[n+11][n+21] = 18 / 6 / h
    M[n+11][n+22] = -9 / 6 / h
    M[n+11][n+23] = 2 / 6 / h

    #
    M[n+12][n+9] = 2 / 6 / h
    M[n+12][n+10] = -9 / 6 / h
    M[n+12][n+11] = 18 / 6 / h
    M[n+12][n+12] = -11 / 6 / h

    M[n+12][n+21] = -11 / 6 / h
    M[n+12][n+22] = 18 / 6 / h
    M[n+12][n+23] = -9 / 6 / h
    M[n+12][n+24] = 2 / 6 / h

    #
    M[n+13][n+10] = 2 / 6 / h
    M[n+13][n+11] =  -9 / 6 / h
    M[n+13][n+12] = 18 / 6 / h
    M[n+13][n+13] = -11 / 6 / h

    M[n+13][n+22] = -11 / 6 / h
    M[n+13][n+23] = 18 / 6 / h
    M[n+13][n+24] = -9 / 6 / h
    M[n+13][n+25] = 2 / 6 / h

    #
    M[n+14][n+11] = 2 / 6 / h
    M[n+14][n+12] = -9 / 6 / h
    M[n+14][n+13] = 18 / 6 / h
    M[n+14][n+14] = -11 / 6 / h

    M[n+14][n+23] = -11 / 6 / h
    M[n+14][n+24] = 18 / 6 / h
    M[n+14][n+25] = -9 / 6 / h
    M[n+14][n+26] = 2 / 6 / h

    #
    M[n+15][n+12] =  2 / 6 / h
    M[n+15][n+13] = -9 / 6 / h
    M[n+15][n+14] = 18 / 6 / h
    M[n+15][n+15] = -11 / 6 / h

    M[n+15][n+24] = -11 / 6 / h
    M[n+15][n+25] = 18 / 6 / h
    M[n+15][n+26] = -9 / 6 / h
    M[n+15][n+27] = 2 / 6 / h
    
    #
    M[n+16][n+19] = 2 / 6 / h
    M[n+16][n+18] = -9 / 6 / h
    M[n+16][n+17] = 18 / 6 / h
    M[n+16][n+16] = -11 / 6 / h

    M[n+16][n+7] = -11 / 6 / h
    M[n+16][n+6] = 18 / 6 / h
    M[n+16][n+5] = -9 / 6 / h
    M[n+16][n+4] = 2 / 6 / h

    # FD

    for i in range(n+17,n+nodes3+15):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Boundary
    n += nodes3

    M[n+15][n+15] = 11 *  h * fFD / 3
    M[n+15][n+14] =  -18 * h * fFD  / 3
    M[n+15][n+13] = 9 * h * fFD / 3
    M[n+15][n+12] = -2 * h * fFD / 3

    return M

markers = ['s','o','x','.']

delta = 1 / float(factor)
vmax = 3./2. * delta * delta 
print("{:.7f}".format(vmax))

# Case 1  
h = delta / 2
nodes1 = int(0.75/h)+1
nodes2 = int(1.25/h)+1
nodes3 = int(1/h) + 1
nodesFull = 3 * nodes3-2

print(nodesFull,h)
x1 = np.linspace(0,0.75,nodes1)
x2 = np.linspace(0.75-2*h,2+2*h,nodes2+4)
x3 = np.linspace(2,3.,nodes3)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull)
forceCoupled = forceCoupling(nodes1+nodes2+nodes3,x,2)

forceCoupled[nodes1-1] = 0
forceCoupled[nodes1] = 0
forceCoupled[nodes1+1] = 0
forceCoupled[nodes1+2] = 0

forceCoupled[nodes1+nodes2+1] = 0
forceCoupled[nodes1+nodes2+2] = 0
forceCoupled[nodes1+nodes2+3] = 0
forceCoupled[nodes1+nodes2+4] = 0

uFDMVHM = solve(Coupling(nodes1,nodes2,nodes3,h),forceCoupled)
uSlice = np.array(np.concatenate((uFDMVHM[0:nodes1],uFDMVHM[nodes1+3:nodes1+nodes2+2],uFDMVHM[nodes1+nodes2+5:len(x)])))

plt.axvline(x=0.75,c="#536872")
plt.axvline(x=2,c="#536872")

if case == "Exact" :

    plt.plot(xFull,uSlice-exactSolution(xFull),c="black",label="m=2",marker=markers[0],markevery=16)

else: 

    
    uFD =  solve(FDM(nodesFull,h),forceFull(nodesFull,h))
    plt.plot(xFull,uSlice-uFD,c="black",label="m=2",marker=markers[0],markevery=16)
    print("h=",h,"m=2",(max(uSlice-uFD)-vmax)/vmax,"{:.7f}".format(max(uSlice-uFD)))

# Case 2
h = delta / 4
nodes1 = int(0.75/h)+1
nodes2 = int(1.25/h)+1
nodes3 = int(1/h) + 1
nodesFull = 3 * nodes3-2

print(nodesFull,h)
x1 = np.linspace(0,0.75,nodes1)
x2 = np.linspace(0.75-4*h,2+4*h,nodes2+8)
x3 = np.linspace(2,3.,nodes3)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull)

forceCoupled = forceCoupling(nodes1+nodes2+nodes3,x,4)

forceCoupled[nodes1-1] = 0
forceCoupled[nodes1] = 0
forceCoupled[nodes1+1] = 0
forceCoupled[nodes1+2] = 0
forceCoupled[nodes1+3] = 0


forceCoupled[nodes1+nodes2+4] = 0
forceCoupled[nodes1+nodes2+5] = 0
forceCoupled[nodes1+nodes2+6] = 0
forceCoupled[nodes1+nodes2+7] = 0
forceCoupled[nodes1+nodes2+8] = 0

uFDMVHM = solve(Coupling4(nodes1,nodes2,nodes3,h),forceCoupled)
uSlice = np.array(np.concatenate((uFDMVHM[0:nodes1-1],uFDMVHM[nodes1+4:nodes1+nodes2+4],uFDMVHM[nodes1+nodes2+9:nodes1+nodes2+nodes3+8])))

if case == "Exact" :

    plt.plot(xFull,uSlice-exactSolution(xFull),c="black",label="m=4",marker=markers[1],markevery=32)

else :

    uFD =  solve(FDM(nodesFull,h),forceFull(nodesFull,h))
    plt.plot(xFull,uSlice-uFD,c="black",label="m=4",marker=markers[1],markevery=32)
    print("h=",h,"m=4",(max(uSlice-uFD)-vmax)/vmax,"{:.7f}".format(max(uSlice-uFD)))

# Case 3
h = delta / 8
nodes1 = int(0.75/h)+1
nodes2 = int(1.25/h)+1
nodes3 = int(1/h) + 1
nodesFull = 3 * nodes3-2

print(nodesFull,h)
x1 = np.linspace(0,0.75,nodes1)
x2 = np.linspace(0.75-8*h,2+8*h,nodes2+16)
x3 = np.linspace(2,3.,nodes3)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull)

forceCoupled = forceCoupling(nodes1+nodes2+nodes3,x,8)

forceCoupled[nodes1-1] = 0
forceCoupled[nodes1] = 0
forceCoupled[nodes1+1] = 0
forceCoupled[nodes1+2] = 0
forceCoupled[nodes1+3] = 0
forceCoupled[nodes1+4] = 0
forceCoupled[nodes1+5] = 0
forceCoupled[nodes1+6] = 0
forceCoupled[nodes1+7] = 0

forceCoupled[nodes1+nodes2+8] = 0
forceCoupled[nodes1+nodes2+9] = 0
forceCoupled[nodes1+nodes2+10] = 0
forceCoupled[nodes1+nodes2+11] = 0
forceCoupled[nodes1+nodes2+12] = 0
forceCoupled[nodes1+nodes2+13] = 0
forceCoupled[nodes1+nodes2+14] = 0
forceCoupled[nodes1+nodes2+15] = 0
forceCoupled[nodes1+nodes2+16] = 0
 
uFDMVHM = solve(Coupling8(nodes1,nodes2,nodes3,h),forceCoupled)
uSlice = np.array(np.concatenate((uFDMVHM[0:nodes1-1],uFDMVHM[nodes1+8:nodes1+nodes2+8],uFDMVHM[nodes1+nodes2+17:nodes1+nodes2+nodes3+16])))

if case == "Exact" :

    plt.plot(xFull,uSlice-exactSolution(xFull),c="black",label="m=8",marker=markers[2],markevery=64)

else :

    uFD =  solve(FDM(nodesFull,h),forceFull(nodesFull,h))
    plt.plot(xFull,uSlice-uFD,c="black",label="m=8",marker=markers[2],markevery=64)
    print("h=",h,"m=8",(max(uSlice-uFD)-vmax)/vmax,"{:.7f}".format(max(uSlice-uFD)))


plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.5f')) 
plt.title("Example with "+example.lower()+" solution for MSCM with $\delta=1/$"+str(factor))
plt.legend()
plt.grid()
plt.xlabel("$x$")

if case == "Exact" :

    plt.ylabel("Error in displacement w.r.t. exact solution")
    plt.savefig("coupling-"+example.lower()+"-approach-2-convergence-exact-moving-"+factor+".pdf",bbox_inches='tight')

else :

    plt.ylabel("Error in displacement w.r.t. FDM")
    plt.savefig("coupling-"+example.lower()+"-approach-2-convergence-fdm-moving-"+factor+".pdf",bbox_inches='tight')