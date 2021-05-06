import math
from scipy.linalg import lapack
import numpy as np
import matplotlib.pyplot as plt
import pickle 

def where(mask, a, b):
    return np.where(mask, a, b)

def solve_implicit(ks, a, b, c, d, b_edge=None, d_edge=None):
    land_mask = (ks >= 0)[:, :, np.newaxis]
    edge_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis])
    water_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :] >= ks[:, :, np.newaxis])

    # print('land mask checksum:', water_mask.sum())
    # print('edge mask checksum:', water_mask.sum())
    # print('water mask checksum:', water_mask.sum())

    #Water masks are created (in a shitty was! represented as doubles instead of bools. increadibly bad)

    a_tri = water_mask * a * np.logical_not(edge_mask)
    b_tri = where(water_mask, b, 1.)
    if b_edge is not None:
        b_tri = where(edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    if d_edge is not None:
        d_tri = where(edge_mask, d_edge, d_tri)

    # print(f'a_tri checksum (right before solve): {a_tri.sum()}')
    # print(f'b_tri checksum (right before solve): {b_tri.sum()}')
    # print(f'c_tri checksum (right before solve): {c_tri.sum()}')
    # print(f'd_tri checksum (right before solve): {d_tri.sum()}')

    return solve_tridiag(a_tri, b_tri, c_tri, d_tri), water_mask


def solve_tridiag(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    a[..., 0] = c[..., -1] = 0  # remove couplings between slices
    return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[3].reshape(a.shape)

def pad_z_edges(arr):
    arr_shape = list(arr.shape)
    arr_shape[2] += 2
    out = np.zeros(arr_shape, arr.dtype)
    out[:, :, 1:-1] = arr
    return out

def _calc_cr(rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return where(vel > 0., rjm, rjp) / where(np.abs(rj) < eps, eps, rj)

def limiter(cr):
    return np.maximum(0., np.maximum(np.minimum(1., 2 * cr), np.minimum(2., cr)))

def _adv_superbee(vel, var, mask, dx, axis, cost, cosu, dt_tracer):
    velfac = 1
    if axis == 0:
        sm1, s, sp1, sp2 = ((slice(1 + n, -2 + n or None), slice(2, -2), slice(None)) for n in range(-1, 3))
        dx = cost[np.newaxis, 2:-2, np.newaxis] * \
            dx[1:-2, np.newaxis, np.newaxis]
    elif axis == 1:
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(1 + n, -2 + n or None), slice(None))
                            for n in range(-1, 3))
        dx = (cost * dx)[np.newaxis, 1:-2, np.newaxis]
        velfac = cosu[np.newaxis, 1:-2, np.newaxis]
    elif axis == 2:
        vel, var, mask = (pad_z_edges(a) for a in (vel, var, mask))
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None))
                            for n in range(-1, 3))
        dx = dx[np.newaxis, np.newaxis, :-1]
    else:
        raise ValueError('axis must be 0, 1, or 2')
    
    uCFL = np.abs(velfac * vel[s] * dt_tracer / dx)
    print(uCFL.shape)
    
    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(rjp, rj, rjm, vel[s]))
    
    return velfac * vel[s] * (var[sp1] + var[s]) * 0.5 - np.abs(velfac * vel[s]) * ((1. - cr) + uCFL * cr) * rj * 0.5

u = np.load('./u.npy')
v = np.load('./v.npy')
w = np.load('./w.npy')
maskU = np.load('./maskU.npy')
maskV = np.load('./maskV.npy')
maskW = np.load('./maskW.npy')
dxt = np.load('./dxt.npy')
dxu = np.load('./dxu.npy')
dyt = np.load('./dyt.npy')
dyu = np.load('./dyu.npy')
dzt = np.load('./dzt.npy')
dzw = np.load('./dzw.npy')
cost = np.load('./cost.npy')
cosu = np.load('./cosu.npy')
kbot = np.load('./kbot.npy')
kappaM = np.load('./kappaM.npy')
mxl = np.load('./mxl.npy')
forc = np.load('./forc.npy')
forc_tke_surface = np.load('./forc_tke_surface.npy')
tke = np.load('./tke.npy')
dtke = np.load('./dtke.npy')

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

b_tri[:, :, 1:-1] = (delta[:, :, 1:-1] + delta[:, :, :-2])
b_tri[:, :, 1:-1] /= dzw[np.newaxis, np.newaxis, 1:-1] 
b_tri[:, :, 1:-1] += 1       
b_tri[:, :, 1:-1] += dt_tke * c_eps * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1]

b_tri[:, :, -1] = 1 + delta[:, :, -2] / (0.5 * dzw[-1])
b_tri[:, :, -1] += dt_tke * c_eps / mxl[2:-2, 2:-2, -1] * sqrttke[2:-2, 2:-2, -1]

b_tri_edge = 1 + delta / dzw[np.newaxis, np.newaxis, :] \
        + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]

c_tri[:, :, :-1] = -delta[:, :, :-1] / dzw[np.newaxis, np.newaxis, :-1]

d_tri[...] = tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :] #dt_tke is one.

d_tri[:, :, -1] += dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1])

sol, water_mask = solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge = b_tri_edge)

print('tke just before where', tke.sum())

tke[2:-2, 2:-2, :, taup1] = np.where(water_mask, sol, tke[2:-2, 2:-2, :, taup1])
tke_surf_corr = np.zeros(maskU.shape[:2])

print('tke after where', tke.sum())


mask = tke[2:-2, 2:-2, -1, taup1] < 0.0
tke_surf_corr[2:-2, 2:-2] = where(
    mask,
    -tke[2:-2, 2:-2, -1, taup1] * 0.5 * dzw[-1] / dt_tke,
    0.
)

tke[2:-2, 2:-2, -1, taup1] = np.maximum(0., tke[2:-2, 2:-2, -1, taup1])

flux_east[:-1, :, :] = K_h_tke * (tke[1:, :, :, tau] - tke[:-1, :, :, tau]) \
    / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) * maskU[:-1, :, :]
flux_east[-1, :, :] = 0

flux_north[:, :-1, :] = K_h_tke * (tke[:, 1:, :, tau] - tke[:, :-1, :, tau]) \
    / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis]
flux_north[:, -1, :] = 0.

print('after maximum before crazy tke', tke.sum())

tke[2:-2, 2:-2, :, taup1] += dt_tke * maskW[2:-2, 2:-2, :] * \
    ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
        + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
        / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))

print('before adv unroll tke', tke.sum())

maskUtr = np.zeros_like(maskW)
maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]
flux_east[...] = 0.
flux_east[1:-2, 2:-2, :] = _adv_superbee(u[..., tau], tke[:, :, :, tau], maskUtr, dxt, 0, cost, cosu, dt_tracer)

maskVtr = np.zeros_like(maskW)
maskVtr[:, :-1, :] = maskW[:, 1:, :] * maskW[:, :-1, :]
flux_north[...] = 0.
flux_north[2:-2, 1:-2, :] = _adv_superbee(v[..., tau], tke[:, :, :, tau], maskVtr, dyt, 1, cost, cosu, dt_tracer)

maskWtr = np.zeros_like(maskW)
maskWtr[:, :, :-1] = maskW[:, :, 1:] * maskW[:, :, :-1]
flux_top[...] = 0.
flux_top[2:-2, 2:-2, :-1] = _adv_superbee(w[..., tau], tke[:, :, :, tau], maskWtr, dzw, 2, cost, cosu, dt_tracer)

dtke[2:-2, 2:-2, :, tau] = maskW[2:-2, 2:-2, :] * (\
      ((flux_east[1:-3, 2:-2, :] - flux_east[2:-2, 2:-2, :]) / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])) - \
      (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :]) / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])
      )
    
dtke[:, :, 0, tau] -= flux_top[:, :, 0] / dzw[0]
dtke[:, :, 1:-1, tau] += - (flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1]
dtke[:, :, -1, tau] += - (flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1])

print('before last tke', tke.sum())

tke[:, :, :, taup1] +=  (1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1]

print('end tke', tke.sum())

# print(f'sqrttke checksum: {sqrttke.sum()}')
# print(f'a_tri checksum: {a_tri.sum()}')
# print(f'b_tri checksum: {b_tri.sum()}')
# print(f'b_tri_edge checksum: {b_tri_edge.sum()}')
# print(f'c_tri checksum: {c_tri.sum()}')
# print(f'd_tri checksum: {d_tri.sum()}')
# print(f'delta checksum: {delta.sum()}')
# print(f'sol {sol.sum()}')
print(f'tke {tke.sum()}')
# print(f'tke surf corr {tke_surf_corr.sum()}')
# print(f'flux east {tke_surf_corr.sum()}')
# print(f'dtke checksum: {dtke.sum()}')
# print(f'flux_east checksum: {flux_east.sum()}')
# print(f'flux_north checksum: {flux_north.sum()}')
# print(f'flux_top checksum: {flux_top.sum()}')


"""



#HERE WE HIJACK SOL TO BE THE SOLUTION OBTAINED FROM ERRONOUS GTSV KERNEL

# with open('./numpy_files/sol.pickle', 'rb') as file:
#     sol = pickle.load(file).reshape((28, 28, 4))



tracer * ((1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1])


print(f'delta checksum: {delta.sum()}')
print(f'a_tri checksum: {a_tri.sum()}')
print(f'b_tri checksum: {b_tri.sum()}')
print(f'c_tri checksum: {c_tri.sum()}')
print(f'd_tri checksum: {d_tri.sum()}')
print(f'sol checksum: {sol.sum()}')
print(f'tke checksum: {tke.sum()}')
print(f'tke_ surf corr checksum: {tke_surf_corr.sum()}')
print(f'flux_top checksum: {flux_top.sum()}')


"""
