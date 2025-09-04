from openeye import oechem
from openeye.oechem import *
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--replicate-lambda-dir' , help = 'replicate#/lambda_#/Production_MD')
args = parser.parse_args()

#First, initialize an OpenEye "molecule stream" to read from a file
istream = oemolistream("../../../lig_156.sdf")
#Create a molecule object
mol = oechem.OEGraphMol()
#Read the molecule
OEReadMolecule(istream, mol)
#Close file
istream.close()


#MDAnalysis universe creation
topology = f'{args.replicate_lambda_dir}/new.gro'
trajectory = f'{args.replicate_lambda_dir}/traj.xtc'
u = mda.Universe(topology, trajectory)
ligand = u.select_atoms('resname LIG and resid 1')
print('There are {} atoms in the ligands'.format(len(ligand.atoms)))

pos_1 = []
pos_2 = []
pos_3 = []
pos_4 = []

nbor_bgn = []
nbor_end = []

#iterate through the rotatable bonds in the molecule
for bond in mol.GetBonds(IsRotor()):
    bgn_atom = bond.GetBgn()
    pos_2 = bond.GetBgnIdx()
    end_atom = bond.GetEnd()
    pos_3 = bond.GetEndIdx()
    #save the bgn bond atom and end bond atom to a list
    #find the neighbors of the atoms from there lists
    print(f"Rotatable bond:" , bond.GetBgnIdx(), "," , bond.GetEndIdx())
    nbors_pos2 = [atom.GetIdx() for atom in bgn_atom.GetAtoms()]
    #nbor_bgn.append(nbor1)
    #print(nbors_pos1)
    for atom in nbors_pos2:
        if atom != pos_3:
            pos_1 = atom
            break
    nbors_pos3 = [atom.GetIdx() for atom in end_atom.GetAtoms()]
    for atom in nbors_pos3:
        if atom != pos_2:
            pos_4 = atom
            break
    print(pos_1)
    print(pos_2)
    print(pos_3)
    print(pos_4)
    torsion = [pos_1, pos_2, pos_3, pos_4]
    print(torsion)
    ag = mda.AtomGroup(ligand.atoms[(torsion)])
    ags = [ag]
    Run = Dihedral(ags).run()
    shape = (Run.angles.shape)
    results = Run.results.angles
    frames = [range(0,1001)]
    replicate_num, lambda_num,_ = tuple(args.replicate_lambda_dir.split("/"))
    image_name = f'{replicate_num}_{lambda_num}'

    plt.scatter(frames, results, color= 'black', alpha=0.5)
    #plt.title(f'{image_name}')
    plt.title( f'Dihedral {torsion} for {image_name}')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    #plt.show()
    plt.savefig(f'dihedral_switch_plots/lig156_{torsion}_{image_name}_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    plt.hist(results, edgecolor="k", color="w")
    plt.title( f'Dihedral {torsion} for {image_name}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frame Count / 1000')
    #plt.show()
    plt.savefig(f'dihedral_switch_plots/lig156_{torsion}_{image_name}_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.close()
