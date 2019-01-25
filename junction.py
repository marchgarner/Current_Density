#This code was tested using Python 3.6.4, GPAW 1.3.0, and ASE 3.16.0

from ase.io import read,write
from ase.visualize import view
import numpy as np
from ase.build import *
from copy import deepcopy
from ase.constraints import FixAtoms
import subprocess
import os.path
from ase.io.trajectory import PickleTrajectory
from matplotlib import pylab as plt
from ase import Atoms

def Junc(Mol, path, A, D1, D2, Alkyne):
	
	#Sulfur and Nitrogen(NH2) are used interchangeably
	mol = Mol
	
	##Define vacuum indices
	#Sulfur_1
	Me1_ind = 0
	S1_ind = 4
	C1_ind = 5
	Si1_ind = 6
	
	#Sulfur_2
	Me2_ind = len(mol)-7
	S2_ind = len(mol)-3
	C2_ind = len(mol)-2
	Si2_ind = len(mol)-1
	

	###basic setup###
	#make slab
	SH_dist = 1.75 #Use 1.5 for NH2
	HH_dist = 0.75

	#allign S-C with y-axis
	CCvec = mol[C1_ind].position - mol[C2_ind].position
	mol.rotate(CCvec,'z')

	a = A
	d1 = D1
	
	H1a = Atoms(['H'],[mol[S1_ind].position+[0,0,SH_dist]])
	H1b = Atoms(['H'],[mol[S1_ind].position+[0,0,SH_dist+HH_dist]])
	mol = mol + H1a + H1b
	mol.set_angle(C1_ind,S1_ind,-2,A)
	mol.set_angle(C1_ind,S1_ind,-1,A)
	if Alkyne is False:
		mol.set_dihedral(Si1_ind,C1_ind,S1_ind,-2, d1)
		mol.set_dihedral(Si1_ind,C1_ind,S1_ind,-1, d1)
	else:
		for n in np.arange(0,360,0.1):
			old1 = mol.get_distance(h1a,-1)
			old2 = mol.get_distance(h1b,-1)
			mol.set_dihedral(Si1_ind,C1_ind,S1_ind,-2, n)
			mol.set_dihedral(Si1_ind,C1_ind,S1_ind,-1, n)
			
			if mol.get_distance(h1a,-1) > mol.get_distance(h1b,-1) and old1 < old2:
				break
			print('dihedral is ', n)

	d2 = D2
	
	H2a = Atoms(['H'],[mol[S2_ind].position+[0,0,SH_dist]])
	H2b = Atoms(['H'],[mol[S2_ind].position+[0,0,SH_dist+HH_dist]])
	mol = mol + H2a + H2b
	mol.set_angle(C2_ind,S2_ind,-2,A)
	mol.set_angle(C2_ind,S2_ind,-1,A)
	if Alkyne is False:
		mol.set_dihedral(Si2_ind,C2_ind,S2_ind,-2, d2)
		mol.set_dihedral(Si2_ind,C2_ind,S2_ind,-1, d2)
	else:
		for n in np.arange(0,360,0.1):
			old1 = mol.get_distance(h2a,-1)
			old2 = mol.get_distance(h2b,-1)
			mol.set_dihedral(Si2_ind,C2_ind,S2_ind,-2, n)
			mol.set_dihedral(Si2_ind,C2_ind,S2_ind,-1, n)
			
			if mol.get_distance(h2a,-1) > mol.get_distance(h2b,-1) and old1 < old2:
				break
			print('dihedral is ', n)

	#set cell og periodic boundary conditions
	mol.center(vacuum=4.0)



	path2 = os.getcwd()
	folder = path
	write('%s/hh_junc.traj'%(folder),mol)#

	return mol

