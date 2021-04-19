# Generate test data and answers for numpy equivalent functions

import glob
import os
import numpy as np
import shutil
from copy import copy
import subprocess
from scipy.linalg import lapack

def generate_fixed_kernel_code(num_args, kernel_name, test_tag, size):
    kernel_type = {1: 'run_1d_kernel',
                   2: 'run_broadcast_kernel', 
                   3: 'run_where_kernel'}[num_args]

    XYZIO = ' '.join(['\t\t{X, Y, Z,},' for i in range(num_args + 1)])
    ShapeIO = ' '.join(['\t\t{0, 0, 0,},' for i in range(num_args + 1)])
    kernel_args = '\n'.join([XYZIO] + [ShapeIO] * 2)

    with open(f'./templates/host_template_{test_tag}.cpp', 'r') as file:
            template = file.read()

    host_code = copy(template)
    kernel_code = kernel_run_code(kernel_type, f'{kernel_name}4d', kernel_args)
    
    for string, s in zip(['X', 'Y', 'Z'], size):
        host_code = host_code.replace(f'<{string}>', str(s))

    host_code = host_code.replace('<func>', kernel_name)
    host_code = host_code.replace('<convertFixed>', '\n\t'.join(
        [f'data_in* arg{i}_fixed = aligned_alloc<data_in>(std::stoi(size));' for i in range(num_args)]
    ))

    host_code = host_code.replace('<loadArgs>', '\n\t'.join(
        [f'xt::xarray<double> arg{i} = xt::load_npy<double>("./npfiles/{kernel_name}_arg{i}_size" + size + ".npy");' for i in range(num_args)]))
    
    host_code = host_code.replace(f'<convertFixedFill>', '\n\t\t'.join(
        [f'arg{i}_fixed[i] = arg{i}[i];' for i in range(num_args)]
    ))

    host_code = host_code.replace('<args>', ', '.join(
        [f'arg{i}_fixed' for i in range(num_args)]))
    host_code = host_code.replace('<kernelRun>', kernel_code)

    return host_code

def generate_common_kernel_code(num_args, kernel_name, test_tag, size):
    kernel_type = {1: 'run_1d_kernel',
                   2: 'run_broadcast_kernel', 
                   3: 'run_where_kernel'}[num_args]


    XYZIO = ' '.join(['\t\t{X, Y, Z,},' for i in range(num_args + 1)])
    ShapeIO = ' '.join(['\t\t{0, 0, 0,},' for i in range(num_args + 1)])
    kernel_args = '\n'.join([XYZIO] + [ShapeIO] * 2)

    with open(f'./templates/host_template_{test_tag}.cpp', 'r') as file:
            template = file.read()

    host_code = copy(template)
    kernel_code = kernel_run_code(kernel_type, f'{kernel_name}4d', kernel_args)
    
    for string, s in zip(['X', 'Y', 'Z'], size):
        host_code = host_code.replace(f'<{string}>', str(s))

    host_code = host_code.replace('<func>', kernel_name)
    host_code = host_code.replace('<loadArgs>', '\n\t'.join(
        [f'xt::xarray<double> arg{i} = xt::load_npy<double>("./npfiles/{kernel_name}_arg{i}_size" + size + ".npy");' for i in range(num_args)]))
    host_code = host_code.replace('<args>', ', '.join(
        [f'arg{i}.data()' for i in range(num_args)]))
    host_code = host_code.replace('<kernelRun>', kernel_code)

    return host_code

def kernel_run_code(kernel_type, kernel_name, argument_inputs):
    code = [f'{kernel_type}("{kernel_name}", inputs, outputs, ',
            argument_inputs,
            'devices, context, bins, q);']
    return '\n'.join(code)


os.environ["XILINX_XRT"] = "/opt/xilinx/xrt"
os.environ["PATH"] = """/opt/xilinx/xrt/bin:/tools/Xilinx/Vitis/2020.1/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/arm/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-none/bin:/tools/Xilinx/Vitis/2020.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/tps/lnx64/cmake-3.3.2/bin:/tools/Xilinx/Vitis/2020.1/cardano/bin:/tools/Xilinx/Vivado/2020.1/bin:/tools/Xilinx/DocNav:/opt/xilinx/xrt/bin:/tools/Xilinx/Vitis/2020.1/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/arm/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-none/bin:/tools/Xilinx/Vitis/2020.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/tps/lnx64/cmake-3.3.2/bin:/tools/Xilinx/Vitis/2020.1/cardano/bin:/tools/Xilinx/Vivado/2020.1/bin:/tools/Xilinx/DocNav:/home/joachim/.local/bin:/home/joachim/bin:/home/joachim/anaconda3/bin:/home/joachim/anaconda3/condabin:/home/joachim/.local/bin:/home/joachim/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/joachim/.dotnet/tools:/home/joachim/.dotnet/tools:/home/joachim/bin:/home/joachim/.npm-packages/bin:/home/joachim/bin:/home/joachim/.npm-packages/bin"""
os.environ["LD_LIBRARY_PATH"] = "/opt/xilinx/xrt/lib:/opt/xilinx/xrt/lib:"
os.environ["PYTHONPATH"] = "/opt/xilinx/xrt/python:/opt/xilinx/xrt/python:"

test_suites = glob.glob('../*')
test_suites.remove('../generate_test_data')
test_suites.remove('../shared')
test_suites = [folder[3:] for folder in test_suites]

def gtsv_func(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    a[..., 0] = c[..., -1] = 0  # remove couplings between slices
    return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[3].reshape(a.shape)


function_getter = { 
                    'abs': np.abs,
                    'add': np.add,
                    'and': np.logical_and, 
                    'div': np.divide, 
                    'eet': np.equal, 
                    'get': np.greater_equal,
                    'gt': np.greater,
                    'gtsv': gtsv_func,
                    'max': np.maximum, 
                    'min': np.minimum,
                    'mult': np.multiply, 
                    'not': np.logical_not,
                    'sqrt': np.sqrt, 
                    'sub': np.subtract, 
                    'where': np.where,
                    }

def generate_pragma_pipeline_kernel_code(*args, **kwargs):
    with open('./templates/host_pragma_pipeline.cpp') as file:
        data = file.read()
    return data

host_code_gen_getter = {
                    'baseline': generate_common_kernel_code,
                    'fixed_point': generate_fixed_kernel_code,
                    'pragma_pipeline': generate_pragma_pipeline_kernel_code,
}

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
    'gtsv': 4,
}

preprocess = {
    'add': lambda x: x,
    'max': lambda x: x,
    'div': lambda x: x,
    # idk put some ones and zeros are random everywhere and then where them all togethere heheh
    'where': lambda x: np.where(x > np.random.randint(0, x.max(), size=x.shape), 1, 0),
    'sqrt': lambda x: np.abs(x),
    'gt': lambda x: x,
    # create binary variables from random data
    'and': lambda x: np.where(x > x.mean(), 1, 0),
    'abs': lambda x: x,
    'not': lambda x: np.where(x > x.mean(), 1, 0),
    'get': lambda x: x,
    'mult': lambda x: x,
    'sub': lambda x: x,
    'min': lambda x: x,
    'eet': lambda x: x,
    'gtsv': lambda x: x,
}

def return_gtsv_host():
    with open('./templates/host_template_gtsv.cpp') as file:
        data = file.read()
    return data

special_host = {
    'gtsv': return_gtsv_host
}

BASE = '/'.join(os.getcwd().split('/')[:-1])

COMPILE_HOST = False
COMPILE_FPGA = False

factors_of_two = 15
sizes = [[6, 6, 4]]

only_add = True

print(f'found {test_suites}')

for factor in range(factors_of_two):
    old_size = sizes[-1]
    new_sizes = [int(s * 2**(1/3)) for s in old_size]
    sizes.append(new_sizes)

with open('available_sizes', 'w') as file:
    #total_size_str = f'\t{[size[0] * size[1] * size[2]] + [str(s) for s in size]}'
    file.write('\n'.join(['\t'.join([str(s) for s in size] + [str(size[0] * size[1] * size[2])]) for size in sizes]) + '\n')

for test_case in test_suites:
    files = glob.glob(f'../{test_case}/*')
    #files.remove(f'../{test_case}/gtsv')
    func_names = [file.split('/')[-1] for file in files]

    for file in func_names:
        if only_add:
            if file != 'add':
                continue

        os.chdir(f'{BASE}/generate_test_data')
        num_args = argnum[file]

        print(f'doing {test_case}... {file} has {num_args} arguments.., creating correct output... ')
        pre_func = preprocess[file]

        for size in sizes:
            size_int = size[0] * size[1] * size[2]
            input_data = [np.random.rand(*size)*1000 for _ in range(4)]

            args = [pre_func(arg) for arg in input_data[0:num_args]]
            func = function_getter[file]
            correct_out = func(*args).astype('double')

            if not os.path.isdir(f'../{test_case}/{file}/npfiles/'): #check if we need to make an np files dir
                    os.mkdir(f'../{test_case}/{file}/npfiles/')
    
            for i, arg in enumerate(args):
                np.save(f'../{test_case}/{file}/npfiles/{file}_arg{i}_size{size_int}.npy', arg.astype('double'))
            np.save(f'../{test_case}/{file}/npfiles/{file}_result_size{size_int}.npy', correct_out)

        # copy current kernels and other static files needed for compilations
        kernel_path = glob.glob(f'../../src/kernels/{file}*.cpp')[0]
        kernel_file_name = kernel_path.split('/')[-1]
        existing_file_path = f'../{test_case}/{file}/kernels/{kernel_file_name}'

        print(f'checking if file exists..: {existing_file_path}... it {"does" if os.path.isfile(existing_file_path) else "does not"}')  
        if not os.path.isfile(existing_file_path):
            if not os.path.isdir(f'../{test_case}/{file}/kernels'): #create kernels dir if not found...
                os.mkdir(f'../{test_case}/{file}/kernels')
            shutil.copy(kernel_path, existing_file_path)
        shutil.copy(f'./makefiles/Makefile_{test_case}', f'../{test_case}/{file}/Makefile')
        shutil.copy(f'./design.cfg', f'../{test_case}/{file}/')
        shutil.copy(f'./emconfig.json', f'../{test_case}/{file}/')

        if file in special_host:
            host_code = special_host[file]()
        else:
            host_code = host_code_gen_getter[test_case](num_args, kernel_name=file, test_tag=test_case, size=size)

        with open(f'../{test_case}/{file}/host_{test_case}.cpp', 'w') as open_file:
                open_file.write(host_code)

        if COMPILE_HOST: 
                os.chdir(f'{BASE}/{test_case}/{file}')
                subprocess.run(['make', 'host'], stdout=True)
                print(f'host created for {file}')
        
        if COMPILE_FPGA:
                os.chdir(f'{BASE}/{test_case}/{file}')
                subprocess.run(['make'], stdout=True)
                print(f'bin created for {file}')
