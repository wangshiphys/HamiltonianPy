from numpy.linalg import inv

import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.base import base_table
from HamiltonianPy.matrepr import termmatrix
from HamiltonianPy.model import Model, Periodization
from HamiltonianPy.greenfunc import GFED
from HamiltonianPy.lattice import Lattice


def CPT(cluster_info, cell_info, coeff_dict, coeff_generator, omegas, ks, 
        max_neighbor=1, spin_dof=2, orbit_dof=1, occupy=None, numbu=False, 
        elta=0.05):
    
    cluster = Lattice(**cluster_info)
    cell = Lattice(**cell_info)
    
    model = Model(cluster=cluster, spin_dof=spin_dof, orbit_dof=orbit_dof,
                  max_neighbor=max_neighbor, numbu=numbu)
    
    Hterms = model(coeff_dict, coeff_generator)

    periodic = Periodization(cluster=cluster, cell=cell, spin_dof=spin_dof,
                             orbit_dof=orbit_dof, numbu=numbu)

    total_dof = cluster.num * spin_dof * orbit_dof

    gf = GFED(Hterms=Hterms, lAoCs=model.lAoCs, rAoCs=model.rAoCs,
              StateMap=model.StateMap, lAoCMap=model.lAoCMap,
              rAoCMap=model.rAoCMap, dof=total_dof, occupy=occupy, numbu=numbu)

    gf_sys = np.zeros((len(omegas), len(ks)), dtype=np.complex128)
    for i, omega in enumerate(omegas):
        gf_cluster = gf(omega + elta * 1j)
        for j, k in enumerate(ks):
            kb = np.dot(k, cell.bs)
            V = model.perturbation(kb)
            gf_cpt = inv(inv(gf_cluster) - V)
            gf_sys[i, j] = periodic(gf_cpt, kb).trace()
        print("The current omega: {0:.2f}".format(omega))

    
    A = -np.imag(gf_sys) / np.pi / cluster.num * cell.num
    plt.figure()
    plt.contourf(range(len(ks)), omegas, A, 500)
    plt.colorbar()
    plt.show()
