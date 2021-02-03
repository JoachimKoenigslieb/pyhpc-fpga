#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:07:23 2020

@author: joachim
"""

def fill_mem(port, mem_type, files):
    mem_slot, mem_ind, file_pnt = 0, 0, 0
    mem_array = []
    
    while mem_slot < 4:
        while mem_ind < 15:
            file = files[file_pnt]

            mem_array.append(f'sp={file}_1:{port}:{mem_type}[{mem_slot}]')
            mem_ind += 1
            file_pnt += 1
            
            if file_pnt == len(files):
                break    
        if file_pnt == len(files):
            break

        mem_slot += 1
        mem_ind = 0
    
    return mem_array

def connect_simple(input_files):
    kernels_1in = []
    kernels_2in = ['abs', 'add', 'and', 'div', 'eet', 'get', 'gt', 'max', 'min', 'mult', 'not', 'sub', ]
    kernels_3in = ['where']
    kernels_4in = ['gtsv']
    
    mem_ind = 0
    mem_slot = 0
    mem_type='PLRAM'
    file_pnt = 0
    
    mem_slots = [None] * 4
    
    mem_array = []
    
    mem_array += fill_mem('strides_offsets_out', 'PLRAM', input_files)
    mem_array += fill_mem('A', 'DRAM', input_files)    

    
    # print(mem_array)
    # print(len(mem_array))

    return mem_array
    
import glob

files =  glob.glob('kernels/*')
files = [file.split('/')[-1] for file in files]

print('\n'.join(connect_simple(files)))