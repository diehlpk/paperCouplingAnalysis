#!/usr/bin/env python3
# Coupling using the displacement (Problem (17)) for m-convergence
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 02/05/2021
import numpy as np
import sys 
import matplotlib.pyplot as plt
import matplotlib

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
    
    force = np.zeros(3*n+2*m)
   
    for i in range(1,3*n+2*m):
        force[i] = f(x[i])
    
    force[3*n+2*m-1] = g
    
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

    #M[n-1][n-1] = 3*h
    #M[n-1][n-2] = -4*h
    #M[n-1][n-3] = h

    M[n-1][n-1] = 11*h / 3
    M[n-1][n-2] = -18*h / 3
    M[n-1][n-3] = 9 * h / 3
    M[n-1][n-4] = -2 * h / 3


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
        M[i][i-2] = -1.  * fPD
        M[i][i-1] = -4. * fPD
        M[i][i] = 10. * fPD
        M[i][i+1] =  -4. * fPD
        M[i][i+2] = -1. * fPD

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

    #M[3*n+3][3*n+3] = 3*  h * fFD 
    #M[3*n+3][3*n+2] = -4*h * fFD  
    #M[3*n+3][3*n+1] = h * fFD 

    M[3*n+3][3*n+3] = 11 *  h * fFD / 3
    M[3*n+3][3*n+2] =  -18 * h * fFD  / 3
    M[3*n+3][3*n+1] = 9 * h * fFD / 3
    M[3*n+3][3*n] = -2 * h * fFD / 3

    #np.savetxt("foo.csv", M, delimiter=",")

    return M

def Coupling4(n,h):

    M = np.zeros([3*n+8,3*n+8])

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
    M[n-1][n+4] = 1

    M[n][n] = -1
    M[n][n-5] = 1

    M[n+1][n+1] = -1
    M[n+1][n-4] = 1

    M[n+2][n+2] = -1
    M[n+2][n-3] = 1

    M[n+3][n+3] = -1
    M[n+3][n-2] = 1

    #M[n+4][n+4] = -1
    #M[n+4][n-1] = 1

    # PD

    for i in range(n+4,2*n+4):

        M[i][i-4] = -(1./64.)/ 2. / h / h *2
        M[i][i-3] =  -(1./24.)/ 2. / h / h*2
        M[i][i-2] = -(1./16.)/ 2. / h / h*2
        M[i][i-1] = -(1./8.)/ 2. / h / h*2
        M[i][i] = (47./96.)/ 2. / h / h*2
        M[i][i+1] =  -(1./8.)/ 2. / h / h*2
        M[i][i+2] = -(1./16.)/ 2. / h / h*2
        M[i][i+3] =  -(1./24.)/ 2. / h / h*2
        M[i][i+4] = -(1./64.)/ 2. / h / h *2

    # Overlap

    #M[2*n+3][2*n+3] = -1
    #M[2*n+3][2*n+8] = 1

    M[2*n+4][2*n+4] = -1
    M[2*n+4][2*n+9] = 1

    M[2*n+5][2*n+5] = -1
    M[2*n+5][2*n+10] = 1

    M[2*n+6][2*n+6] = -1
    M[2*n+6][2*n+11] = 1

    M[2*n+7][2*n+7] = -1
    M[2*n+7][2*n+12] = 1

    M[2*n+8][2*n+8] = -1
    M[2*n+8][2*n+3] = 1

    # FD

    for i in range(2*n+9,3*n+7):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Boundary

    #M[3*n+3][3*n+3] = 3*  h * fFD 
    #M[3*n+3][3*n+2] = -4*h * fFD  
    #M[3*n+3][3*n+1] = h * fFD 

    M[3*n+7][3*n+7] = 11 *  h * fFD / 3
    M[3*n+7][3*n+6] =  -18 * h * fFD  / 3
    M[3*n+7][3*n+5] = 9 * h * fFD / 3
    M[3*n+7][3*n+4] = -2 * h * fFD / 3

    np.savetxt("foo.csv", M, delimiter=",")

    #plt.matshow(M)
    #plt.show()

    return M


def Coupling8(n,h):

    M = np.zeros([3*n+16,3*n+16])

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
    M[n-1][n+8] = 1

    M[n][n] = -1
    M[n][n-9] = 1

    M[n+1][n+1] = -1
    M[n+1][n-8] = 1

    M[n+2][n+2] = -1
    M[n+2][n-7] = 1

    M[n+3][n+3] = -1
    M[n+3][n-6] = 1

    M[n+4][n+4] = -1
    M[n+4][n-5] = 1

    M[n+5][n+5] = -1
    M[n+5][n-4] = 1

    M[n+6][n+6] = -1
    M[n+6][n-3] = 1

    M[n+7][n+7] = -1
    M[n+7][n-2] = 1

    # PD

    for i in range(n+8,2*n+8):
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

    M[2*n+8][2*n+8] = -1
    M[2*n+8][2*n+17] = 1

    M[2*n+9][2*n+9] = -1
    M[2*n+9][2*n+18] = 1

    M[2*n+10][2*n+10] = -1
    M[2*n+10][2*n+19] = 1

    M[2*n+11][2*n+11] = -1
    M[2*n+11][2*n+20] = 1

    M[2*n+12][2*n+12] = -1
    M[2*n+12][2*n+21] = 1

    M[2*n+13][2*n+13] = -1
    M[2*n+13][2*n+22] = 1

    M[2*n+14][2*n+14] = -1
    M[2*n+14][2*n+23] = 1

    M[2*n+15][2*n+15] = -1
    M[2*n+15][2*n+24] = 1

    M[2*n+16][2*n+16] = -1
    M[2*n+16][2*n+7] = 1

    # FD

    for i in range(2*n+17,3*n+15):
        M[i][i-1] = -2 * fFD
        M[i][i] = 4 * fFD
        M[i][i+1] = -2 * fFD

    # Boundary

    #M[3*n+3][3*n+3] = 3*  h * fFD 
    #M[3*n+3][3*n+2] = -4*h * fFD  
    #M[3*n+3][3*n+1] = h * fFD 

    M[3*n+15][3*n+15] = 11 *  h * fFD / 3
    M[3*n+15][3*n+14] =  -18 * h * fFD  / 3
    M[3*n+15][3*n+13] = 9 * h * fFD / 3
    M[3*n+15][3*n+12] = -2 * h * fFD / 3

    np.savetxt("foo.csv", M, delimiter=",")

    #plt.matshow(M)
    #plt.show()

    return M


markers = ['s','o','x','.']


delta = 0.125

# Case 1  
h = delta / 2
nodes = int(1 / h) + 1
nodesFull = 3 * nodes

print(nodes,h)
x1 = np.linspace(0,1,nodes)
x2 = np.linspace(1-2*h,2+2*h,nodes+4)
x3 = np.linspace(2,3.,nodes)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull-2)
forceCoupled = forceCoupling(nodes,x,2)

forceCoupled[nodes-1] = 0
forceCoupled[nodes] = 0
forceCoupled[nodes+1] = 0

forceCoupled[2*nodes+2] = 0
forceCoupled[2*nodes+3] = 0
forceCoupled[2*nodes+4] = 0

uFDMVHM = solve(Coupling(nodes,h),forceCoupled)
uSlice = np.array(np.concatenate((uFDMVHM[0:nodes],uFDMVHM[nodes+3:2*nodes+2],uFDMVHM[2*nodes+5:len(x)])))

plt.plot(xFull,uSlice,c="black",label="m=2",marker=markers[0],markevery=5)

# Case 2
h = delta / 4
nodes = int(1 / h) + 1
nodesFull = 3 * nodes

x1 = np.linspace(0,1,nodes)
x2 = np.linspace(1-4*h,2+4*h,nodes+8)
x3 = np.linspace(2,3.,nodes)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull-2)
forceCoupled = forceCoupling(nodes,x,4)


forceCoupled[nodes-1] = 0
forceCoupled[nodes] = 0
forceCoupled[nodes+1] = 0
forceCoupled[nodes+2] = 0
forceCoupled[nodes+3] = 0


forceCoupled[2*nodes+4] = 0
forceCoupled[2*nodes+5] = 0
forceCoupled[2*nodes+6] = 0
forceCoupled[2*nodes+7] = 0
forceCoupled[2*nodes+8] = 0

uFDMVHM = solve(Coupling4(nodes,h),forceCoupled)
uSlice = np.array(np.concatenate((uFDMVHM[0:nodes-1],uFDMVHM[nodes+4:2*nodes+4],uFDMVHM[2*nodes+9:3*nodes+8])))

#print(np.concatenate((x[0:nodes-1],x[nodes+4:2*nodes+4],x[2*nodes+9:3*nodes+8])))

plt.plot(xFull,uSlice,c="black",label="m=4",marker=markers[1],markevery=5)

# Case 3
h = delta / 8
nodes = int(1 / h) + 1
nodesFull = 3 * nodes

x1 = np.linspace(0,1,nodes)
x2 = np.linspace(1-8*h,2+8*h,nodes+16)
x3 = np.linspace(2,3.,nodes)
x = np.array(np.concatenate((x1,x2,x3)))

xFull = np.linspace(0,3.,nodesFull-2)
forceCoupled = forceCoupling(nodes,x,8)

forceCoupled[nodes-1] = 0
forceCoupled[nodes] = 0
forceCoupled[nodes+1] = 0
forceCoupled[nodes+2] = 0
forceCoupled[nodes+3] = 0
forceCoupled[nodes+4] = 0
forceCoupled[nodes+5] = 0
forceCoupled[nodes+6] = 0
forceCoupled[nodes+7] = 0

forceCoupled[2*nodes+8] = 0
forceCoupled[2*nodes+9] = 0
forceCoupled[2*nodes+10] = 0
forceCoupled[2*nodes+11] = 0
forceCoupled[2*nodes+12] = 0
forceCoupled[2*nodes+13] = 0
forceCoupled[2*nodes+14] = 0
forceCoupled[2*nodes+15] = 0
forceCoupled[2*nodes+16] = 0
 
uFDMVHM = solve(Coupling8(nodes,h),forceCoupled)
#uSlice = np.array(np.concatenate((uFDMVHM[0:nodes-1],uFDMVHM[nodes:2*nodes-1],uFDMVHM[2*nodes:3*nodes])))

plt.plot(x,uFDMVHM,c="black",label="m=8",marker=markers[2],markevery=5)

plt.title("Example with "+example+" solution for Problem (17)")
plt.legend()
plt.grid()
plt.xlabel("$x$")
plt.ylabel("Error in displacement w.r.t exact solution")

plt.savefig("coupling-"+example.lower()+"-approach-1-convergence-exact.pdf",bbox_inches='tight')

