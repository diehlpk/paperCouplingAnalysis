#!/usr/bin/env python3
# Plot the vmax
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 03/02/2021
import numpy as np
import csv
import sys 
import matplotlib.pyplot as plt
import matplotlib

pgf_with_latex = {"text.usetex": True, "font.size" : 12, "pgf.preamble" : [r'\usepackage{xfrac}'] }

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

def read_csv(filename):

    x = []
    with open(filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
      
        for row in plots:
            x.append(float(row[0]))

    return x



vhm_con = read_csv("con-vhm-neumann.txt")
approach_1_con = read_csv("con-approach-1-neumann.txt")
approach_2_con = read_csv("con-approach-2-neumann.txt")

x = np.linspace(0,1,7)

ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels([r"$1/8$", r"$1/16$", r"$1/32$", r"$1/64$",
                     r"$1/128$",r"$1/256$",r"$1/512$"])
plt.plot(x,approach_1_con,label="MDCM",marker="o",color="black")
plt.grid()
plt.legend()
plt.xlabel("delta")
plt.ylabel("con(M)")
plt.savefig("condition-neumann-MDCM.pdf",bbox_inches='tight')

plt.clf()

ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels([r"$1/8$", r"$1/16$", r"$1/32$", r"$1/64$",
                     r"$1/128$",r"$1/256$",r"$1/512$"])
plt.plot(x,vhm_con,label="VHCM",marker="s",color="black")
plt.plot(x,approach_2_con,label="MSCM",marker="x",color="black")
plt.grid()
plt.legend()
plt.xlabel("delta")
plt.ylabel("con(M)")
plt.savefig("condition-neumann.pdf",bbox_inches='tight')

