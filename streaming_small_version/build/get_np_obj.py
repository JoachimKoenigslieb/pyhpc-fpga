import numpy as np

a = np.load('add_arg0_size4096.npy')
b = np.load('add_arg1_size4096.npy')

"""
c, d = [np.random.rand(16, 16, 16) for _ in range(2)]
np.save('add_arg0_size4096.npy', c) 
np.save('add_arg1_size4096.npy', d) 
np.save('add_result_size4096.npy', (c+d) * c)
"""
