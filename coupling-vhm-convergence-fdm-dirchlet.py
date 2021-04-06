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
factor = sys.argv[2]

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

    M[3*n-1][3*n-1] = 1
    
    return M

def CouplingFDVHM4(n,h):

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

    # Node with one neighbor
    M[n+1][n] = -8 * fVHM
    M[n+1][n+1] = 16 * fVHM
    M[n+1][n+2] = -8 * fVHM

    # Node with two neighbor
    M[n+2][n] = -1 * fVHM
    M[n+2][n+1] = -4 * fVHM
    M[n+2][n+2] = 10. * fVHM 
    M[n+2][n+3] = -4 * fVHM
    M[n+2][n+4] = -1 * fVHM

    # Node with three neighbor
    M[n+3][n] = -(1./27.)/ 2. / h / h  *2
    M[n+3][n+1] = -(1./9.)/ 2. / h / h *2
    M[n+3][n+2] = -(2./9.)/ 2. / h / h *2
    M[n+3][n+3] = (20./27.)/ 2. / h / h *2
    M[n+3][n+4] = -(2./9.)/ 2. / h / h *2
    M[n+3][n+5] = -(1./9.)/ 2. / h / h *2
    M[n+3][n+6] = -(1./27.)/ 2. / h / h *2   

    for i in range(n+4,2*n-4):
        M[i][i-4] = -(1./64.)/ 2. / h / h*2
        M[i][i-3] = -(1./24.)/ 2. / h / h *2
        M[i][i-2] = -(1./16.)/ 2. / h / h  *2
        M[i][i-1] = -(1./8.)/ 2. / h / h *2
        M[i][i] = (47./96.)/ 2. / h / h *2
        M[i][i+1] = -(1./8.)/ 2. / h / h  *2
        M[i][i+2] = -(1./16.)/ 2. / h / h *2
        M[i][i+3] = -(1./24.)/ 2. / h / h *2
        M[i][i+4] = -(1./64.)/ 2. / h / h*2

    # Node with three neighbors
    M[2*n-4][2*n-7] = -(1./27.)/ 2. / h / h *2
    M[2*n-4][2*n-6] = -(1./9.)/ 2. / h / h *2
    M[2*n-4][2*n-5] = -(2./9.)/ 2. / h / h *2
    M[2*n-4][2*n-4] = (20./27.)/ 2. / h / h *2
    M[2*n-4][2*n-3] = -(2./9.)/ 2. / h / h *2
    M[2*n-4][2*n-2] = -(1./9.)/ 2. / h / h *2
    M[2*n-4][2*n-1] = -(1./27.)/ 2. / h / h *2

    # Node with two neighbors
    M[2*n-3][2*n-1] = -1 * fVHM
    M[2*n-3][2*n-2] = -4 * fVHM
    M[2*n-3][2*n-3] = 10. * fVHM 
    M[2*n-3][2*n-4] = -4 * fVHM
    M[2*n-3][2*n-5] = -1 * fVHM

    # Node with one neighbor
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

    M[3*n-1][3*n-1] = 1
    
    return M

def CouplingFDVHM8(n,h):

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

    # Node with one neighbor
    M[n+1][n] = -8 * fVHM
    M[n+1][n+1] = 16 * fVHM
    M[n+1][n+2] = -8 * fVHM

    # Node with two neighbor
    M[n+2][n] = -1 * fVHM
    M[n+2][n+1] = -4 * fVHM
    M[n+2][n+2] = 10. * fVHM 
    M[n+2][n+3] = -4 * fVHM
    M[n+2][n+4] = -1 * fVHM

    # Node with three neighbor
    M[n+3][n] = -(1./27.)/ 2. / h / h  *2
    M[n+3][n+1] = -(1./9.)/ 2. / h / h *2
    M[n+3][n+2] = -(2./9.)/ 2. / h / h *2
    M[n+3][n+3] = (20./27.)/ 2. / h / h *2
    M[n+3][n+4] = -(2./9.)/ 2. / h / h *2
    M[n+3][n+5] = -(1./9.)/ 2. / h / h *2
    M[n+3][n+6] = -(1./27.)/ 2. / h / h *2   

    # Node with four neighbors
    M[n+4][n] = -(1./64.)/ 2. / h / h *2
    M[n+4][n+1] = -(1./24.)/ 2. / h / h *2
    M[n+4][n+2] = -(1./16.)/ 2. / h / h *2
    M[n+4][n+3] = -(1./8.)/ 2. / h / h *2
    M[n+4][n+4] = (47./96.)/ 2. / h / h *2
    M[n+4][n+5] = -(1./8.)/ 2. / h / h *2
    M[n+4][n+6] = -(1./16.)/ 2. / h / h *2
    M[n+4][n+7] = -(1./24.)/ 2. / h / h*2
    M[n+4][n+8] = -(1./64.)/ 2. / h / h*2

    # Node with 5 neighbors
    M[n+5][n] = -(1/125) / 2. / h / h*2
    M[n+5][n+1] = -(1/50) / 2. / h / h*2
    M[n+5][n+2] = -(2/75) / 2. / h / h*2
    M[n+5][n+3] = -(1/25) / 2. / h / h*2
    M[n+5][n+4] = -(2/25) / 2. / h / h*2
    M[n+5][n+5] =  (131/375) / 2. / h / h*2
    M[n+5][n+6] =  -(2/25) / 2. / h / h*2
    M[n+5][n+7] =  -(1/25) / 2. / h / h*2
    M[n+5][n+8] = -(2/75) / 2. / h / h *2
    M[n+5][n+9] = -(1/50) / 2. / h / h *2
    M[n+5][n+10] = -(1/125) / 2. / h / h *2

    # Node with 6 neighbors
    M[n+6][n+6-6] =  -(1/216) / 2. / h / h *2
    M[n+6][n+6-5] =  -(1/90) / 2. / h / h *2
    M[n+6][n+6-4] =  -(1/72) / 2. / h / h *2
    M[n+6][n+6-3] =  -(1/54) / 2. / h / h *2
    M[n+6][n+6-2] =  -(1/36) / 2. / h / h *2
    M[n+6][n+6-1] =  -(1/18) / 2. / h / h *2
    M[n+6][n+6] =  (71/270) / 2. / h / h *2
    M[n+6][n+6+1] =  -(1/18) / 2. / h / h *2
    M[n+6][n+6+2] =  -(1/36) / 2. / h / h *2
    M[n+6][n+6+3] =  -(1/54) / 2. / h / h *2
    M[n+6][n+6+4] =  -(1/72) / 2. / h / h *2
    M[n+6][n+6+5] =  -(1/90) / 2. / h / h *2
    M[n+6][n+6+6] =  -(1/216) / 2. / h / h *2

    # Node with 7 neighbors
    M[n+7][n+7-7] = -(1/343)  / 2. / h / h*2
    M[n+7][n+7-6] = -(1/147)  / 2. / h / h*2
    M[n+7][n+7-5] = -(2/245)  / 2. / h / h*2
    M[n+7][n+7-4] = -(1/98)  / 2. / h / h*2
    M[n+7][n+7-3] =  -(2/147)  / 2. / h / h*2
    M[n+7][n+7-2] =  -(1/49)  / 2. / h / h*2
    M[n+7][n+7-1] = -(2/49)  / 2. / h / h*2
    M[n+7][n+7] =  (353/1715)  / 2. / h / h*2
    M[n+7][n+7+1] = -(2/49)  / 2. / h / h*2
    M[n+7][n+7+2] =  -(1/49)  / 2. / h / h*2
    M[n+7][n+7+3] =  -(2/147)  / 2. / h / h*2
    M[n+7][n+7+4] = -(1/98)  / 2. / h / h*2
    M[n+7][n+7+5] = -(2/245)  / 2. / h / h*2
    M[n+7][n+7+6] = -(1/147)  / 2. / h / h*2
    M[n+7][n+7+7] = -(1/343)  / 2. / h / h*2


    for i in range(n+8,2*n-8):
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

    # Node with 7 neighbors
    M[2*n-8][2*n-15] = -(1/343)  / 2. / h / h *2
    M[2*n-8][2*n-14] = -(1/147)  / 2. / h / h*2
    M[2*n-8][2*n-13] = -(2/245)  / 2. / h / h*2
    M[2*n-8][2*n-12] = -(1/98)  / 2. / h / h*2
    M[2*n-8][2*n-11] =  -(2/147)  / 2. / h / h*2
    M[2*n-8][2*n-10] =  -(1/49)  / 2. / h / h*2
    M[2*n-8][2*n-9] =  -(2/49)  / 2. / h / h*2
    M[2*n-8][2*n-8] = (353/1715)  / 2. / h / h*2
    M[2*n-8][2*n-7] = -(2/49)  / 2. / h / h*2
    M[2*n-8][2*n-6] =  -(1/49)  / 2. / h / h*2
    M[2*n-8][2*n-5] = -(2/147)  / 2. / h / h*2
    M[2*n-8][2*n-4] = -(1/98)  / 2. / h / h*2
    M[2*n-8][2*n-3] = -(2/245)  / 2. / h / h*2
    M[2*n-8][2*n-2] = -(1/147)  / 2. / h / h*2
    M[2*n-8][2*n-1] = -(1/343)  / 2. / h / h*2
    
    # Node with 6 neighbors
    M[2*n-7][2*n-13] =  -(1/216) / 2. / h / h *2
    M[2*n-7][2*n-12] =  -(1/90) / 2. / h / h *2
    M[2*n-7][2*n-11] =  -(1/72) / 2. / h / h *2
    M[2*n-7][2*n-10] =  -(1/54) / 2. / h / h *2
    M[2*n-7][2*n-9] =  -(1/36) / 2. / h / h *2
    M[2*n-7][2*n-8] =  -(1/18) / 2. / h / h *2
    M[2*n-7][2*n-7] = (71/270) / 2. / h / h *2
    M[2*n-7][2*n-6] =   -(1/18) / 2. / h / h *2
    M[2*n-7][2*n-5] =  -(1/36) / 2. / h / h *2
    M[2*n-7][2*n-4] =  -(1/54) / 2. / h / h *2
    M[2*n-7][2*n-3] =   -(1/72) / 2. / h / h *2
    M[2*n-7][2*n-2] =  -(1/90) / 2. / h / h *2
    M[2*n-7][2*n-1] =  -(1/216) / 2. / h / h *2
    
    # Node with 5 neighbors   
    M[2*n-6][2*n-11] = -(1/125) / 2. / h / h *2
    M[2*n-6][2*n-10] = -(1/50) / 2. / h / h*2
    M[2*n-6][2*n-9] = -(2/75) / 2. / h / h*2
    M[2*n-6][2*n-8] = -(1/25) / 2. / h / h*2
    M[2*n-6][2*n-7] = -(2/25) / 2. / h / h*2
    M[2*n-6][2*n-6] = (131/375) / 2. / h / h*2
    M[2*n-6][2*n-5] = -(2/25) / 2. / h / h*2
    M[2*n-6][2*n-4] = -(1/25) / 2. / h / h*2
    M[2*n-6][2*n-3] = -(2/75) / 2. / h / h*2
    M[2*n-6][2*n-2] =-(1/50) / 2. / h / h*2
    M[2*n-6][2*n-1] = -(1/125) / 2. / h / h*2
    
    # Node with four neighbors
    M[2*n-5][2*n-9] = -(1./64.)/ 2. / h / h *2
    M[2*n-5][2*n-8] = -(1./24.)/ 2. / h / h*2
    M[2*n-5][2*n-7] = -(1./16.)/ 2. / h / h*2
    M[2*n-5][2*n-6] = -(1./8.)/ 2. / h / h*2
    M[2*n-5][2*n-5] = (47./96.)/ 2. / h / h*2
    M[2*n-5][2*n-4] = -(1./8.)/ 2. / h / h*2
    M[2*n-5][2*n-3] = -(1./16.)/ 2. / h / h*2
    M[2*n-5][2*n-2] = -(1./24.)/ 2. / h / h*2
    M[2*n-5][2*n-1] = -(1./64.)/ 2. / h / h*2
    
    # Node with three neighbors
    M[2*n-4][2*n-7] = -(1./27.)/ 2. / h / h *2
    M[2*n-4][2*n-6] = -(1./9.)/ 2. / h / h *2
    M[2*n-4][2*n-5] = -(2./9.)/ 2. / h / h *2
    M[2*n-4][2*n-4] = (20./27.)/ 2. / h / h *2
    M[2*n-4][2*n-3] = -(2./9.)/ 2. / h / h *2
    M[2*n-4][2*n-2] = -(1./9.)/ 2. / h / h *2
    M[2*n-4][2*n-1] = -(1./27.)/ 2. / h / h *2

    # Node with two neighbors
    M[2*n-3][2*n-5] = -1 * fVHM
    M[2*n-3][2*n-4] = -4 * fVHM
    M[2*n-3][2*n-3] = 10. * fVHM 
    M[2*n-3][2*n-2] = -4 * fVHM
    M[2*n-3][2*n-1] = -1 * fVHM

    # Node with one neighbor
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

    M[3*n-1][3*n-1] = 1

    return M

markers = ['s','o','x','.']


delta = 1 / float(factor)

vmax = (10./81.)*delta*delta - ((4./243.) * delta * delta * delta * (8+3*delta))
print("{:.7f}".format(vmax))

# Case 1  
h = delta / 2
nodes = int(1 / h) + 1
nodesFull = 3 * nodes - 2

x1 = np.linspace(0,1,nodes)
x2 = np.linspace(1,2.,nodes)
x3 = np.linspace(2,3.,nodes)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull)
forceCoupled = forceCoupling(nodes,x)
forceCoupled[nodes-1] = 0
forceCoupled[nodes] = 0

forceCoupled[2*nodes-1] = 0
forceCoupled[2*nodes] = 0

uFDMVHM = solve(CouplingFDVHM(nodes,h),forceCoupled)
uSlice = np.array(np.concatenate((uFDMVHM[0:nodes-1],uFDMVHM[nodes:2*nodes-1],uFDMVHM[2*nodes:3*nodes])))

uFD =  solve(FDM(nodesFull,h),forceFull(nodesFull,h))

plt.plot(xFull,uSlice-uFD,c="black",label="m=2",marker=markers[0],markevery=8)
print("h=",h,"m=2",(max(uSlice-uFD)-vmax)/vmax,"{:.7f}".format(max(uSlice-uFD)))

# Case 2
h = delta / 4
nodes = int(1 / h) + 1
nodesFull = 3 * nodes-2

x1 = np.linspace(0,1,nodes)
x2 = np.linspace(1,2.,nodes)
x3 = np.linspace(2,3.,nodes)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull)
forceCoupled = forceCoupling(nodes,x)
forceCoupled[nodes-1] = 0
forceCoupled[nodes] = 0

forceCoupled[2*nodes-1] = 0
forceCoupled[2*nodes] = 0

uFDMVHM = solve(CouplingFDVHM4(nodes,h),forceCoupled)
uSlice = np.array(np.concatenate((uFDMVHM[0:nodes-1],uFDMVHM[nodes:2*nodes-1],uFDMVHM[2*nodes:3*nodes])))

uFD =  solve(FDM(nodesFull,h),forceFull(nodesFull,h))

plt.plot(xFull,uSlice-uFD,c="black",label="m=4",marker=markers[1],markevery=16)
print("h=",h,"m=4",abs((max(uSlice-uFD)-vmax)/vmax),"{:.7f}".format(max(uSlice-uFD)))

# Case 3
h = delta / 8
nodes = int(1 / h) + 1
nodesFull = 3 * nodes -2 

x1 = np.linspace(0,1,nodes)
x2 = np.linspace(1,2.,nodes)
x3 = np.linspace(2,3.,nodes)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull)
forceCoupled = forceCoupling(nodes,x)

forceCoupled[nodes-1] = 0
forceCoupled[nodes] = 0

forceCoupled[2*nodes-1] = 0
forceCoupled[2*nodes] = 0

uFDMVHM = solve(CouplingFDVHM8(nodes,h),forceCoupled)
uSlice = np.array(np.concatenate((uFDMVHM[0:nodes-1],uFDMVHM[nodes:2*nodes-1],uFDMVHM[2*nodes:3*nodes])))

uFD =  solve(FDM(nodesFull,h),forceFull(nodesFull,h))

plt.plot(xFull,uSlice-uFD,c="black",label="m=8",marker=markers[2],markevery=32)
print("h=",h,"m=8",(max(uSlice-uFD)-vmax)/vmax,"{:.7f}".format(max(uSlice-uFD)))


plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.6f'))
plt.title("Example with "+example.lower()+" solution for VHCM with $\delta=1/$"+str(factor))
plt.legend()
plt.grid()
plt.xlabel("$x$")
plt.ylabel("Error in displacement w.r.t. FDM")

plt.savefig("coupling-"+example.lower()+"-vhm-convergence-fdm-dirchlet-"+factor+".pdf",bbox_inches='tight')


