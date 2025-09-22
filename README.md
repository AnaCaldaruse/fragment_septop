# fragment_septop
Scripts and input files for calculating binding free energies in fragment-based drug design using the Separated Topologies (SepTop) approach.

It is expected that you have the folllowing installed in order to use the scripts here.

## Installation

How to install septop:
```bash
git clone https://github.com/MobleyLab/SeparatedTopologies.git
#cd into directory
mamba env create -f environment.yml
mamba activate septop

#if above does not work:
mamba create -n septop python=3.9
mamba activate septop
mamba install numpy=1.23 --yes
mamba install openeye-toolkits=2021.1
mamba install -c conda-forge openff-toolkit=0.10.06 --yes
mamba install parmed=4.1 --yes
```bash

How to install pymbar and alchemlyb
```bash
mamba install -c conda-forge pymbar alchemlyb
```bash
