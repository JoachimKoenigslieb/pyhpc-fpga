import get_inputs
import subprocess
import re
import pickle

sizes = [2**s for s in range(8, 28)]

d = {'sizes': sizes, }
tri = []

for s in sizes:
	print(f'running {s}')
	inputs = get_inputs.generate_inputs(s) # also saves input to files...

	res = subprocess.run(['kernprof', '-l', '-v', 'check_inputs.py'], capture_output=True)
	o = res.stdout.decode('utf-8').split('\n')[74].strip()
	print(o)
	r = float(re.split("[ ]+", o)[4])
	tri.append(r)

d['tri'] = tri
with open('tri-ratio.p', 'wb') as file:
	pickle.dump(d, file)

