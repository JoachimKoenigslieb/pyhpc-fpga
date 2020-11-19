import math
from scipy.linalg import lapack
import numpy as np
import matplotlib.pyplot as plt
import pickle 

def where(mask, a, b):
    return np.where(mask, a, b)

def gen_plots():
    with open('./numpy_files/a_tri.pickle', 'rb') as file:
        a_tri_fpga = pickle.load(file)
    
    with open('./numpy_files/b_tri.pickle', 'rb') as file:
        b_tri_fpga = pickle.load(file)
    
    with open('./numpy_files/c_tri.pickle', 'rb') as file:
        c_tri_fpga = pickle.load(file)
    
    with open('./numpy_files/d_tri.pickle', 'rb') as file:
        d_tri_fpga = pickle.load(file)
    
    with open('./numpy_files/sol.pickle', 'rb') as file:
        sol_fpga = pickle.load(file)
        
    M_fpga = np.diag(a_tri_fpga[1:], k=-1) + np.diag(b_tri_fpga) + np.diag(c_tri_fpga[:-1], k=1)
    fpga_sol_on_host = solve_tridiag(a_tri_fpga, b_tri_fpga, c_tri_fpga, d_tri_fpga)
    
    f,a = plt.subplots()
    a.plot(fpga_sol_on_host, alpha=1, color='red', label='FPGA tris, but solved in lapack', )
    a.plot(sol.flatten(), linestyle=(0, (5, 10)), color='blue', alpha=1, label='np tris and lapack solution')
    a.plot(sol_fpga, color='green', label='FPGA tris solved w. vitis solver')
    a.legend()
    
    f,a = plt.subplots(2)
    fpga_sol_diff = M_fpga @ sol_fpga - d_tri_fpga
    a[0].plot(fpga_sol_diff)
    a[0].set_title(f'M_fpga @ sol_fpga - d_tri_fpga \nmean: {fpga_sol_diff.mean():0.1f} std: {fpga_sol_diff.std():0.1f}')
    sol_diff = M @ sol.flatten() - d_tri.flatten()
    a[1].plot(sol_diff)
    a[1].set_title(f'M @ sol - d_tri \nmean: {sol_diff.mean():0.3f} std: {sol_diff.std():0.3f}')
    f.suptitle('is it a correct solution?', y=1.05)
    f.tight_layout()
    
    f, a = plt.subplots()
    f.suptitle('cumulative sums of solutions.')
    a.plot(np.cumsum(sol.flatten()), label='np solution')
    a.plot(np.cumsum(sol_fpga), label='fpga solution')
    f.legend()
    
    
    f, a = plt.subplots(2,2)
    f.suptitle('tridagonals and rhs fpga sim vs np', y=1.05)
    a[0,0].plot(np.cumsum(a_tri_fpga), label='fpga')
    a[0,0].plot(np.cumsum(a_tri.flatten()), linestyle='-.', label='np')
    a[0,0].set_title('a_tri')
    
    a[0,1].plot(np.cumsum(b_tri_fpga))
    a[0,1].plot(np.cumsum(b_tri.flatten()), linestyle='-.')
    a[0,1].set_title('b_tri')
    
    a[1,0].plot(np.cumsum(c_tri_fpga))
    a[1,0].plot(np.cumsum(c_tri.flatten()), linestyle='-.')
    a[1,0].set_title('c_tri')
    
    a[1,1].plot(np.cumsum(d_tri_fpga))
    a[1,1].plot(np.cumsum(d_tri.flatten()), linestyle='-.')
    a[1,1].set_title('d_tri (rhs)')
    
    f.tight_layout()
    f.legend()


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

print(f' a tri before "last row": {a_tri.sum()}')
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

print('before masks:')
print(f'sqrttke checksum: {sqrttke.sum()}')
print(f'delta checksum: {delta.sum()}')
print(f'a_tri checksum: {a_tri.sum()}')
print(f'b_tri checksum: {b_tri.sum()}')
print(f'c_tri checksum: {c_tri.sum()}')
print(f'd_tri checksum: {d_tri.sum()}')

land_mask = (ks >= 0)[:, :, np.newaxis]
edge_mask = land_mask & (np.arange(a_tri.shape[2])[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis])
water_mask = land_mask & (np.arange(a_tri.shape[2])[np.newaxis, np.newaxis, :] >= ks[:, :, np.newaxis])

a_tri = water_mask * a_tri * np.logical_not(edge_mask)
b_tri = np.where(water_mask, b_tri, 1.)
b_tri = np.where(edge_mask, b_tri_edge, b_tri)
c_tri = water_mask * c_tri
d_tri = water_mask * d_tri

sol, water_mask = solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge = b_tri_edge)

tke_surf_corr = np.zeros(maskU.shape[:2])

#HERE WE HIJACK SOL TO BE THE SOLUTION OBTAINED FROM ERRONOUS GTSV KERNEL

with open('./numpy_files/sol.pickle', 'rb') as file:
    sol = pickle.load(file)
    sol = sol.reshape((28, 28, 4))

tmp_tke = np.where(water_mask, sol, tke[2:-2, 2:-2, :, taup1])
tke[2:-2, 2:-2, :, taup1] = tmp_tke

mask = tke[2:-2, 2:-2, -1, taup1] < 0.0


print(f'sqrttke checksum: {sqrttke.sum()}')
print(f'delta checksum: {delta.sum()}')
print(f'a_tri checksum: {a_tri.sum()}')
print(f'b_tri checksum: {b_tri.sum()}')
print(f'c_tri checksum: {c_tri.sum()}')
print(f'd_tri checksum: {d_tri.sum()}')
print(f'sol checksum: {sol.sum()}')
print(f'tke checksum: {tke.sum()}')
