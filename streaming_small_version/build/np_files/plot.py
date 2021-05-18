import pickle
import matplotlib.pyplot as plt

with open('tri-ratio.p', 'rb') as file:
	d = pickle.load(file)
    
with open('tri-ratio-threadripper.p', 'rb') as file:
	d_threadripper = pickle.load(file)

s, r = d.values()    
s_t, r_t = d_threadripper.values()

plt.close('all')

f, a = plt.subplots(1, 1, figsize=(15, 8))
a.plot(s, r, 'o--', label='Laptop')
a.plot(s_t, r_t, 'x--', label='Threadripper')
a.semilogx(base=10)
a.set_xlabel('# of sites in benchmark')
a.set_ylabel('% of time spent in LAPACK solver')

a.legend()