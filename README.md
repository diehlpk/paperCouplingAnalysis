# Supplementary material: Coupling approaches for classical linear elasticity and bond-basedperidynamic models

[![DOI](https://zenodo.org/badge/336118235.svg)](https://zenodo.org/badge/latestdoi/336118235) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/ca0978c2b61949f292aa4663d67e1115)](https://www.codacy.com/gh/diehlpk/paperCouplingAnalysis/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=diehlpk/paperCouplingAnalysis&amp;utm_campaign=Badge_Grade)

## Dependencies

* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)

## Files

* coupling-vhm.py - VHCM
* coupling-vhm-convergence.py - VHCM (m-convergence)
* coupling-vhm-direchlet.py - VHCM (direchlet)
* coupling-approach-1.py - MDCM
* coupling-approcah-1-convergence.py - MDCM (m-convergence)
* coupling-approach-1-direchlet.py - MDCM (dirichlet) 
* coupling-approach-2.py - MSCM 
* coupling-approach-2-dirichlet.py - MSCM (dirichlet) 
* coupling-approach-2-convergence.py - MSCM (m-convergnece)

## Usage

```bash
python3 -m venv deps
source deps/bin/activate
pip install -r requirements.txt

