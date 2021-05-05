import math
import numpy as np


def generate_inputs(size):
    np.random.seed(17)

    # shape = (
    #     math.ceil(2 * size ** (1/3)),
    #     math.ceil(2 * size ** (1/3)),
    #     math.ceil(0.25 * size ** (1/3)),
    # )
    
    shape = (6, 6, 6)


    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw = (np.random.randn(shape[2]) for _ in range(2))

    # 2d arrays
    kbot = np.random.randint(0, shape[2], size=shape[:2]).astype('float64')
    forc_tke_surface = np.random.randn(*shape[:2])

    # 3d arrays
    kappaM, mxl, forc = (np.random.randn(*shape) for _ in range(3))
    maskU, maskV, maskW = ((np.random.rand(*shape) < 0.8).astype('float64') for _ in range(3))

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

N = 128
u, v, w , maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, kbot, kappaM, mxl, forc, forc_tke_surface, tke, dtke = generate_inputs(N)

np.save('./u', u)
np.save('./v', v)
np.save('./w', w)
np.save('./maskU', maskU)
np.save('./maskV', maskV)
np.save('./maskW', maskW)
np.save('./dxt', dxt)
np.save('./dxu', dxu)
np.save('./dyt', dyt)
np.save('./dyu', dyu)
np.save('./dzt', dzt)
np.save('./dzw', dzw)
np.save('./cost',cost)
np.save('./cosu', cosu)
np.save('./kbot', kbot)
np.save('./kappaM', kappaM)
np.save('./mxl', mxl)
np.save('./forc', forc)
np.save('./forc_tke_surface', forc_tke_surface)
np.save('./tke', tke)
np.save('./dtke', dtke)

