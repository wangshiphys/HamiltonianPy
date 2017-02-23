import numpy as np

__all__ = ['lattice_constructor','chain8_info', 'chain16_info', 'chain24_info',
           'kpath']

def lattice_constructor(n):# {{{
    p0 = np.array([[0]])
    v = np.array([[1]])
    buff = []
    for i in range(n):
        buff.append(p0 + i * v)
    points = np.concatenate(buff, axis=0)
    tvs = n * v
    info = {'points': points, 'tvs': tvs}
    return info
# }}}

chain8_info = lattice_constructor(8)
chain16_info = lattice_constructor(16)
chain24_info = lattice_constructor(24)

def path_constructor(num):# {{{
    ks = np.linspace(-0.5, 0.5, num)
    return ks
# }}}

kpath = path_constructor(201)
