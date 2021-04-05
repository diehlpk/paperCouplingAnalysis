#!/usr/bin/env python3
# Plot the solution of all approaches in one plot
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


markers = ['s','o','x','.']


n = np.power(2,4)
h = 1./n
nodesFull = 3 * n + 1

xFull = np.linspace(0,3.,nodesFull)



uMDCM = np.loadtxt("coupling-"+example.lower()+"-approach-1.csv",)   
uMSCM = np.loadtxt("coupling-"+example.lower()+"-approach-2.csv")    
uVHCM = np.loadtxt("coupling-"+example.lower()+"-vhm.csv")    

plt.plot(xFull,exactSolution(xFull),label="Exact solution",c="black")
plt.plot(xFull,uMDCM,label=r"MDCM",c="black",marker=markers[0],markevery=8)
plt.plot(xFull,uMSCM,label=r"MSCM",c="black",marker=markers[1],markevery=8)
plt.plot(xFull,uVHCM,label=r"VHCM",c="black",marker=markers[2],markevery=8)
plt.ylabel("Displacement")
       
plt.title("Example with "+example.lower()+" solution with $\delta=1/8$ and $m=2$")
plt.legend()
plt.grid()
plt.xlabel("$x$")

plt.savefig("coupling-"+example.lower()+"-all.pdf",bbox_inches='tight')

