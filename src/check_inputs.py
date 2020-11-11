import math
from scipy.linalg import lapack
import numpy as np

def where(mask, a, b):
    return np.where(mask, a, b)


def solve_implicit(ks, a, b, c, d, b_edge=None, d_edge=None):
    land_mask = (ks >= 0)[:, :, np.newaxis]
    edge_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis])
    water_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :] >= ks[:, :, np.newaxis])

    #Water masks are created (in a shitty was! represented as doubles instead of bools. increadibly bad)

    a_tri = water_mask * a * np.logical_not(edge_mask)
    b_tri = where(water_mask, b, 1.)
    if b_edge is not None:
        b_tri = where(edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    if d_edge is not None:
        d_tri = where(edge_mask, d_edge, d_tri)

    return solve_tridiag(a_tri, b_tri, c_tri, d_tri), water_mask


def solve_tridiag(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    a[..., 0] = c[..., -1] = 0  # remove couplings between slices
    return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[3].reshape(a.shape)


u = np.load('./numpy_files/u.npy')
v = np.load('./numpy_files/v.npy')
w = np.load('./numpy_files/w.npy')
maskU = np.load('./numpy_files/maskU.npy')
maskV = np.load('./numpy_files/maskV.npy')
maskW = np.load('./numpy_files/maskW.npy')
dxt = np.load('./numpy_files/dxt.npy')
dxu = np.load('./numpy_files/dxu.npy')
dyt = np.load('./numpy_files/dyt.npy')
dyu = np.load('./numpy_files/dyu.npy')
dzt = np.load('./numpy_files/dzt.npy')
dzw = np.load('./numpy_files/dzw.npy')
cost = np.load('./numpy_files/cost.npy')
cosu = np.load('./numpy_files/cosu.npy')
kbot = np.load('./numpy_files/kbot.npy')
kappaM = np.load('./numpy_files/kappaM.npy')
mxl = np.load('./numpy_files/mxl.npy')
forc = np.load('./numpy_files/forc.npy')
forc_tke_surface = np.load('./numpy_files/forc_tke_surface.npy')
tke = np.load('./numpy_files/tke.npy')
dtke = np.load('./numpy_files/dtke.npy')

tau = 0
taup1 = 1
taum1 = 2

dt_tracer = 1
dt_mom = 1
AB_eps = 0.1
alpha_tke = 1.
c_eps = 0.7
K_h_tke = 2000.

flux_east = np.zeros_like(maskU)
flux_north = np.zeros_like(maskU)
flux_top = np.zeros_like(maskU)

sqrttke = np.sqrt(np.maximum(0., tke[:, :, :, tau])) 


"""
integrate Tke equation on W grid with surface flux boundary condition
"""
dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

"""
vertical mixing and dissipation of TKE
"""
ks = kbot[2:-2, 2:-2] - 1

a_tri = np.zeros_like(maskU[2:-2, 2:-2])
b_tri = np.zeros_like(maskU[2:-2, 2:-2])
c_tri = np.zeros_like(maskU[2:-2, 2:-2])
d_tri = np.zeros_like(maskU[2:-2, 2:-2])
delta = np.zeros_like(maskU[2:-2, 2:-2])


delta[:, :, :-1] = 1 / dzt[np.newaxis, np.newaxis, 1:] * alpha_tke * 0.5 * dt_tke\
        * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:])
a_tri[:, :, 1:-1] = -delta[:, :, :-2] / \
        dzw[np.newaxis, np.newaxis, 1:-1]
a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * dzw[-1])

b_tri[:, :, 1:-1] = 1 + \
		(delta[:, :, 1:-1] + delta[:, :, :-2]) / dzw[np.newaxis, np.newaxis, 1:-1] + \
		dt_tke * c_eps * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1]

b_tri[:, :, -1] = 1 + delta[:, :, -2] / (0.5 * dzw[-1]) \
        + dt_tke * c_eps / mxl[2:-2, 2:-2, -1] * sqrttke[2:-2, 2:-2, -1]

b_tri_edge = 1 + delta / dzw[np.newaxis, np.newaxis, :] \
        + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]

c_tri[:, :, :-1] = -delta[:, :, :-1] / dzw[np.newaxis, np.newaxis, :-1]

d_tri[...] = tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :] #dt_tke is one.
d_tri[:, :, -1] += dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1])

land_mask = (ks >= 0)[:, :, np.newaxis]
edge_mask = land_mask & (np.arange(a_tri.shape[2])[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis])
water_mask = land_mask & (np.arange(a_tri.shape[2])[np.newaxis, np.newaxis, :] >= ks[:, :, np.newaxis])

a_tri = water_mask * a_tri * np.logical_not(edge_mask)
b_tri = np.where(water_mask, b_tri, 1.)
b_tri = np.where(edge_mask, b_tri_edge, b_tri)
c_tri = water_mask * c_tri
d_tri = water_mask * d_tri

sol, water_mask = solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge = b_tri_edge)

print(f'sqrttke checksum: {sqrttke.sum()}')
print(f'delta checksum: {delta.sum()}')
print(f'a_tri checksum: {a_tri.sum()}')
print(f'b_tri checksum: {b_tri.sum()}')
print(f'c_tri checksum: {c_tri.sum()}')
print(f'd_tri checksum: {d_tri.sum()}')
print(f'sol checksum: {sol.sum()}')
