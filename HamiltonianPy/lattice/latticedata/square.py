import numpy as np

__all__ = ['cell_info', 'cluster22_info', 'cluster33_info',
           'cluster23_info', 'cluster32_info', 'cluster34_info', 
           'cluster43_info', 'kpath', 'kmesh']

def lattice_constructor(nx, ny):# {{{
    p0 = np.array([[0, 0]])
    vx = np.array([[1, 0]])
    vy = np.array([[0, 1]])
    buff = []
    for i in range(nx):
        for j in range(ny):
            buff.append(p0 + i * vx + j * vy)
    points = np.concatenate(buff, axis=0)
    tvs = np.concatenate([nx * vx, ny * vy], axis=0)
    info = {'points': points, 'tvs': tvs}
    return info
# }}}

cell_info = lattice_constructor(1, 1)

cluster22_info = lattice_constructor(2, 2)

cluster23_info = lattice_constructor(2, 3)

cluster32_info = lattice_constructor(3, 2)

cluster33_info = lattice_constructor(3, 3) 

cluster43_info = lattice_constructor(4, 3) 

cluster34_info = lattice_constructor(3, 4) 

def path_constructor(num):# {{{
    pathx0 = np.linspace(0, 0.5, num)
    pathy0 = np.zeros((num, ))
    pathx1 = np.ones((num, )) * 0.5
    pathy1 = np.linspace(0, 0.5, num)
    pathx2 = np.linspace(0.5, 0, 2*num)
    pathy2 = np.linspace(0.5, 0, 2*num)
    pathx = np.concatenate((pathx0, pathx1, pathx2))
    pathy = np.concatenate((pathy0, pathy1, pathy2))
    path = list(zip(pathx, pathy))
    return path
# }}}

kpath = path_constructor(101)

def mesh_constructor(numkx, numky):# {{{
    kx = np.linspace(-0.5, 0.5, numkx)
    ky = np.linspace(-0.5, 0.5, numky)
    KX, KY = np.meshgrid(kx, ky)
    ks = list(zip(KX.flat, KY.flat))
    return ks
# }}}

kmesh = mesh_constructor(201, 201)
