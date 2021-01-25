# Generate test data and answers for numpy equivalent functions

import glob 
import os
import numpy as np
import shutil
from copy import copy
import subprocess

files = glob.glob('../*')
files.remove('../generate_test_data')
files.remove('../gtsv')
files.remove('../shared')
files = [file[3:] for file in files]

nptranslater = {'div': 'divide', 'gt': 'greater', 'and': 'logical_and', 'not': 'logical_not', 'get': 'greater_equal', 
	'mult': 'multiply', 'sub': 'subtract', 'eet': 'equal', 'max': 'maximum', 'min': 'minimum'}

argnum = {
	'add': 2,
	'max': 2,
	'div': 2,
	'where': 3,
	'sqrt': 1, 
	'gt': 2,
	'and': 2,
	'abs': 1,
	'not': 1,
	'get': 2,
	'mult': 2,
	'sub': 2,
	'min': 2,
	'eet': 2,
}

preprocess = {
	'add': lambda x: x,
	'max': lambda x: x,
	'div': lambda x: x,
	'where': lambda x: np.where(x>np.random.randint(0, x.max(), size=x.shape), 1, 0), # idk put some ones and zeros are random everywhere and then where them all togethere heheh
	'sqrt': lambda x: np.abs(x),
	'gt': lambda x: x,
	'and': lambda x: np.where(x>x.mean(), 1, 0), # create binary variables from random data
	'abs': lambda x: x,
	'not': lambda x: np.where(x>x.mean(), 1, 0),
	'get': lambda x: x,
	'mult': lambda x: x,
	'sub': lambda x: x,
	'min': lambda x: x,
	'eet': lambda x: x,
}

os.environ["XILINX_XRT"] = "/opt/xilinx/xrt"
os.environ["PATH"] = """/opt/xilinx/xrt/bin:/tools/Xilinx/Vitis/2020.1/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/arm/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-none/bin:/tools/Xilinx/Vitis/2020.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/tps/lnx64/cmake-3.3.2/bin:/tools/Xilinx/Vitis/2020.1/cardano/bin:/tools/Xilinx/Vivado/2020.1/bin:/tools/Xilinx/DocNav:/opt/xilinx/xrt/bin:/tools/Xilinx/Vitis/2020.1/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/arm/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-none/bin:/tools/Xilinx/Vitis/2020.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/tps/lnx64/cmake-3.3.2/bin:/tools/Xilinx/Vitis/2020.1/cardano/bin:/tools/Xilinx/Vivado/2020.1/bin:/tools/Xilinx/DocNav:/home/joachim/.local/bin:/home/joachim/bin:/home/joachim/anaconda3/bin:/home/joachim/anaconda3/condabin:/home/joachim/.local/bin:/home/joachim/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/joachim/.dotnet/tools:/home/joachim/.dotnet/tools:/home/joachim/bin:/home/joachim/.npm-packages/bin:/home/joachim/bin:/home/joachim/.npm-packages/bin"""
os.environ["LD_LIBRARY_PATH"] = "/opt/xilinx/xrt/lib:/opt/xilinx/xrt/lib:"
os.environ["PYTHONPATH"] = "/opt/xilinx/xrt/python:/opt/xilinx/xrt/python:"

BASE = '/home/joachim/Documents/speciale/vitis/veros-port/tests'

npfuncs = {func_name: np.__dict__[func_name if func_name not in nptranslater else nptranslater[func_name]] for func_name in files}

COMPILE_HOST = False
COMPILE_FPGA = False

input_data = [np.random.rand(6, 6, 4)*1000 for _ in range(3)]

def generate_common_kernel_code(num_args, kernel_name):
	kernel_type = {1: 'run_1d_kernel', 2: 'run_broadcast_kernel', 3: 'run_where_kernel'}[num_args]

	XYZIO = ' '.join(['\t\t{X, Y, Z,},' for i in range(num_args + 1)])
	ShapeIO = ' '.join(['\t\t{0, 0, 0,},' for i in range(num_args + 1)])
	kernel_args = '\n'.join([XYZIO] + [ShapeIO] * 2)

	host_code = copy(template)
	kernel_code = kernel_run_code(kernel_type, f'{kernel_name}4d', kernel_args)

	host_code = host_code.replace('<func>', kernel_name)
	host_code = host_code.replace('<loadArgs>', '\n\t'.join([f'xt::xarray<double> arg{i} = xt::load_npy<double>("{kernel_name}_arg{i}.npy");' for i in range(num_args)]))
	host_code = host_code.replace('<args>', ', '.join([f'arg{i}.data()' for i in range(num_args)]))
	host_code = host_code.replace('<kernelRun>', kernel_code)

	return host_code

def kernel_run_code(kernel_type, kernel_name, argument_inputs):
	code = [f'{kernel_type}("{kernel_name}", inputs, outputs, ', 
			argument_inputs,
			'devices, context, bins, q);']
	return '\n'.join(code)

with open('./host_template.cpp', 'r') as file:
	template = file.read()

for file in files:
	num_args = argnum[file]

	print(f'{file} has {num_args} arguments.., creating correct output...')
	pre_func = preprocess[file]

	args = [pre_func(arg) for arg in input_data[0:num_args]]
	func = npfuncs[file]
	correct_out = func(*args).astype('double')

	for i, arg in enumerate(args):
		np.save(f'../{file}/{file}_arg{i}.npy', arg.astype('double'))

	np.save(f'../{file}/{file}_result.npy', correct_out)

	#copy current kernels and other static files needed for compilations
	shutil.copy(f'../../src/kernels/{file}4d.cpp', f'../{file}/kernels/{file}4d.cpp')
	shutil.copy(f'./Makefile', f'../{file}/')
	shutil.copy(f'./design.cfg', f'../{file}/')
	shutil.copy(f'./emconfig.json', f'../{file}/')

	host_code = generate_common_kernel_code(num_args, file)

	with open(f'../{file}/host.cpp', 'w') as open_file:
		open_file.write(host_code)

	os.chdir(f'{BASE}/{file}')
	if COMPILE_HOST: 
		subprocess.run(['make', 'host'], stdout=True)
		print(f'host created for {file}')
	
	if COMPILE_FPGA:
		subprocess.run(['make'], stdout=True)
		print(f'bin created for {file}')

	

print(host_code)



