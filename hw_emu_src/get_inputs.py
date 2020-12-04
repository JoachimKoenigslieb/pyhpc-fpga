import math
import numpy as np


def generate_inputs(size):
    np.random.seed(17)

    shape = (
        math.ceil(2 * size ** (1/3)),
        math.ceil(2 * size ** (1/3)),
        math.ceil(0.25 * size ** (1/3)),
    )

    # masks
    maskU, maskV, maskW = ((np.random.rand(*shape) < 0.8).astype('float64') for _ in range(3))

    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw = (np.random.randn(shape[2]) for _ in range(2))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))

    # 2d arrays
    kbot = np.random.randint(0, shape[2], size=shape[:2]).astype('float64')
    forc_tke_surface = np.random.randn(*shape[:2])

    # 3d arrays
    kappaM, mxl, forc = (np.random.randn(*shape) for _ in range(3))

    # 4d arrays
    u, v, w, tke, dtke = (np.random.randn(*shape, 3) for _ in range(5))

    
    return (
        u, v, w,
        maskU, maskV, maskW,
        dxt, dxu, dyt, dyu, dzt, dzw,
        cost, cosu,
        kbot,
        kappaM, mxl, forc,
        forc_tke_surface,
        tke, dtke
    )

N = 4096
u, v, w , maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, kbot, kappaM, mxl, forc, forc_tke_surface, tke, dtke = generate_inputs(N)

np.save('./numpy_files/u', u)
np.save('./numpy_files/v', v)
np.save('./numpy_files/w', w)
np.save('./numpy_files/maskU', maskU)
np.save('./numpy_files/maskV', maskV)
np.save('./numpy_files/maskW', maskW)
np.save('./numpy_files/dxt', dxt)
np.save('./numpy_files/dxu', dxu)
np.save('./numpy_files/dyt', dyt)
np.save('./numpy_files/dyu', dyu)
np.save('./numpy_files/dzt', dzt)
np.save('./numpy_files/dzw', dzw)
np.save('./numpy_files/cost',cost)
np.save('./numpy_files/cosu', cosu)
np.save('./numpy_files/kbot', kbot)
np.save('./numpy_files/kappaM', kappaM)
np.save('./numpy_files/mxl', mxl)
np.save('./numpy_files/forc', forc)
np.save('./numpy_files/forc_tke_surface', forc_tke_surface)
np.save('./numpy_files/tke', tke)
np.save('./numpy_files/dtke', dtke)

