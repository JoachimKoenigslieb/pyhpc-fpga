#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:39:06 2020

@author: joachim
"""

import numpy as np
import random
import itertools

def one_pad(A, length):
    #length is the total length we want
    return tuple(1 for _ in range(length - len(A))) + A

def zero_pad(A, length):
    return tuple(0 for _ in range(length - len(A))) + A

def get_lin_index(ind, strides, offset, lin_offset):
    return sum([(i+off)*s for i,s,off in zip(ind, strides, offset)]) + lin_offset

def broadcast(A, B):
    #one-pad shapes. (add empty dimensions so their equal sized!)
    max_dim = max(len(A), len(B))
    A, B = one_pad(A, max_dim), one_pad(B, max_dim)
    
    broadcasted = tuple(max(a,b) if (a==1 or b==1 or a==b) else False for a, b in zip(A,B))
    if False in broadcasted:
        raise TypeError(f'Cant broadcast shapes {A} with {B}')
    return broadcasted

def negotiate_shapes(out, *inputs):
    #out is shape-tuple, inouts is lsit of shape tuples
    #first, we check broadcasting compatability.
    for i, input_shape in enumerate(inputs[:-1]):
        inputs_broadcasted_shape = broadcast(input_shape, inputs[i+1])
    if tuple(i for i in inputs_broadcasted_shape if i!=1) != tuple(i for i in out if i!=1):
        raise TypeError(f'Cant place inputs {inputs} into out. Broadcasted input shape is {inputs_broadcasted_shape}')

def view_shape(shape, start, end):
    return tuple(sh-st+e for sh,st,e in zip(shape, start,end))

def stride_from_shape(A):
    stride=[1]
    for dim in reversed(A[1:]):
        if dim != 1:
            stride.append(dim*stride[-1])
        else:
            stride.append(0)
    return tuple(reversed(stride))

def collect_linear_offsets(view_shapes, strides, offsets):
    linear_offsets = []
    for shape, stride, offset in zip(view_shapes, strides, offsets):
        linear_offsets.append(sum([st * off if sh == 1 else 0 for sh, st, off in zip(shape, stride, offset)]))
    return linear_offsets

def rebuild_strides(strides_in, view_shapes_in, out_dim):
    strides_in = [tuple(st for st, sh in zip(stride, shape) if sh != 1) for stride, shape in zip(strides_in, view_shapes_in)] #filter out any singleton dimensions
    strides_in = [zero_pad(stride, out_dim) for stride in strides_in] #zero pad if need to
    return strides_in

def build_offsets(start_out, starts_in, view_shapes_in, out_dim, max_dim):
    #filter starts in
    #subtract
    #gg
    
    starts_in_singleton_filtered = [tuple(st for st, sh in zip(start, shape) if sh != 1) for start, shape in zip(starts_in, view_shapes_in)]
    offsets = [tuple(st_in - st_out for st_in, st_out in zip(start_in, start_out)) for start_in in starts_in_singleton_filtered]
    return offsets

def negotiate_strides(test_case):
    shapes_in, shape_out = test_case["shapes"][:-1], test_case["shapes"][-1] 
    starts_in, start_out = test_case["starts"][:-1], test_case["starts"][-1]
    ends_in, end_out = test_case["ends"][:-1], test_case["ends"][-1]

    view_shapes_in, view_shape_out = [view_shape(inp, start,end) for inp, start, end in zip(shapes_in, starts_in, ends_in)], view_shape(shape_out, start_out, end_out)

    negotiate_shapes(view_shape_out, *view_shapes_in) #make sure that it can actaully broadcast
    
    strides_in, stride_out = [stride_from_shape(sh) for sh in shapes_in], stride_from_shape(shape_out)
    linear_offsets = collect_linear_offsets(view_shapes_in, strides_in, starts_in)

    out_dim = len(shape_out)
    max_dim = max([len(shape_out)] + [len(sh) for sh in shapes_in])
    
    strides_in = rebuild_strides(strides_in, view_shapes_in, out_dim)
    
    offsets = build_offsets(start_out, starts_in, view_shapes_in, out_dim, max_dim)
    out_indeces = itertools.product(*[range(st, sh+end) for st, sh, end in zip(start_out, shape_out, end_out)])
    
    # for ind in out_indeces:
        # lin_index_inputs = [get_lin_index(ind, stride_in, offset, linear_offset) for stride_in, offset, linear_offset in zip(strides_in, offsets, linear_offsets)]
        # lin_index_output = get_lin_index(ind, stride_out, [0]*out_dim, 0)
        # print(f'{ind}. out: {lin_index_output} ' + ' '.join([f'input_{i}: {lin_index}' for i, lin_index in enumerate(lin_index_inputs)]))

    print(f"""Negoiated strides...
          input strides: {strides_in} output_strides: {stride_out}
          input offsets: {offsets} input lin offset: {linear_offsets}""")

    return strides_in, stride_out, offsets, linear_offsets, out_indeces


def run_test(test_case):
    #first, we build random data based on the shapes:
    inputs = [np.random.randint(0, 100, shape) for shape in test_case["shapes"][:-1]]
    starts_in, starts_out = test_case["starts"][:-1], test_case["starts"][-1]
    ends_in, ends_out = test_case["ends"][:-1], test_case["ends"][-1]
    
    input_slices = [tuple(slice(st, end if end !=0 else None) for st, end in zip(starts, ends)) for starts, ends in zip(starts_in, ends_in)]
    output_slice = tuple(slice(st, end if end !=0 else None) for st, end in zip(starts_out, ends_out))
    
    output = np.zeros(test_case["shapes"][-1])
    output[output_slice] = np.squeeze(inputs[0][input_slices[0]]) + np.squeeze(inputs[1][input_slices[1]])
    
    strides_in, stride_out, offsets, linear_offsets, out_indeces = negotiate_strides(test_case)
    
    A_strides, B_strides = strides_in
    A_offset, B_offset = offsets
    A_lin_offset, B_lin_offset = linear_offsets
    out_dim = len(stride_out)
    
    inputs_flat = [inp.flatten() for inp in inputs]
    output_flat = np.zeros_like(output.flatten())
    
    # print(list(out_indeces))
    
    for ind in out_indeces:
        A_ind, B_ind = get_lin_index(ind, A_strides, A_offset, A_lin_offset), get_lin_index(ind, B_strides, B_offset, B_lin_offset)
        A_val = inputs_flat[0][A_ind]
        B_val = inputs_flat[1][B_ind]
        out_ind = get_lin_index(ind, stride_out, [0]*out_dim, 0)
        print(f'{ind}: out_ind: {out_ind}. A_ind: {A_ind}, B_ind: {B_ind}. A_val: {A_val}, B_val: {B_val}')
        output_flat[out_ind] = A_val + B_val
        
    print(f'Homerolled sum: {output_flat.sum()}')
    print(f'Numpy sum: {output.sum()}')
    
    print(output)
    print(output_flat.reshape(output.shape))
    

test_cases = [
    {"shapes": [(7, 7, 3), (1,), (5, 5,)],
     "starts": [(2, 1, 0), (0,), (1, 0)],
     "ends": [(-2, -5, 0), (0,), (-1, -2)]}
    ]

run_test(test_cases[0])