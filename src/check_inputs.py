import math
import numpy as np

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

print(b_tri.sum())
