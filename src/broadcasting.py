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

def squeeze(arr):
    #squeezing an array remvoes all ones!
    return tuple(i for i in arr if i!= 1)

def zero_on_squeeze(arr, view_shape):
    return tuple(a if vs != 1 else 0 for a, vs in zip(arr, view_shape))

def negotiate_shapes(out, *inputs):
    #out is shape-tuple, inouts is lsit of shape tuples
    #first, we check broadcasting compatability.
    squeezed_inputs = [squeeze(inp) for inp in inputs]
    for i, input_shape in enumerate(squeezed_inputs[:-1]):
        inputs_broadcasted_shape = broadcast(input_shape, squeezed_inputs[i+1])
    if tuple(i for i in inputs_broadcasted_shape if i!=1) != squeeze(out):
        raise TypeError(f'Cant place inputs {inputs} into out. Broadcasted input shape is {inputs_broadcasted_shape}')

def view_shape(shape, start, end):
    return tuple(sh-st+e for sh,st,e in zip(shape, start,end))

def stride_from_shape(A):
    stride=[1]
    
    for dim in reversed(A[1:]):
        stride.append(dim*stride[-1])

    return tuple(reversed(stride))

def collect_linear_offsets(view_shapes, strides, offsets):
    linear_offsets = []
    for shape, stride, offset in zip(view_shapes, strides, offsets):
        linear_offsets.append(sum([st * off if sh == 1 else 0 for sh, st, off in zip(shape, stride, offset)]))
    return linear_offsets

def rebuild_strides(strides_in, view_shapes_in, out_dim):
    strides_in = [zero_on_squeeze(stride, view_shape) for stride, view_shape in zip(strides_in, view_shapes_in)]
    strides_in = [zero_pad(stride, out_dim) for stride in strides_in] #zero pad if need to 
    #actually dont zero pad! put zeros into the places where view_shapes_in has ones!
    
    return strides_in

def build_offsets(start_out, starts_in, view_shapes_in, out_dim):
    starts_in_singleton_filtered = [zero_on_squeeze(start, view_shape) for start, view_shape in zip(starts_in, view_shapes_in)]
    starts_in_singleton_filtered = [zero_pad(filtered_start, out_dim) for filtered_start in starts_in_singleton_filtered]
    offsets = [tuple(st_in - st_out for st_in, st_out in zip(start_in, start_out)) for start_in in starts_in_singleton_filtered]
    return offsets

def negotiate_strides(test_case, debug=True):
    #read in values.
    shapes_in, shape_out = test_case["shapes"][:-1], test_case["shapes"][-1] 
    starts_in, start_out = test_case["starts"][:-1], test_case["starts"][-1]
    ends_in, end_out = test_case["ends"][:-1], test_case["ends"][-1]
    if debug: print(f'Shapes in: {shapes_in}. shapes out: {shape_out}')

    #calculate view shapes
    view_shapes_in, view_shape_out = [view_shape(inp, start,end) for inp, start, end in zip(shapes_in, starts_in, ends_in)], view_shape(shape_out, start_out, end_out)
    if debug: print(f'View shapes in: {view_shapes_in} out: {view_shape_out}')

    #make sure shapes can be broadcast!
    # negotiate_shapes(view_shape_out, *view_shapes_in) #make sure that it can actaully broadcast
    
    #get strides. these are fundementally dependend on the data shape!
    strides_in, stride_out = [stride_from_shape(sh) for sh in shapes_in], stride_from_shape(shape_out)
    if debug: print(f'strides in: {strides_in} out: {stride_out}')
    
    #collect linear offsets. This means taking into account the starting offsets from "collapsed" dimensions.
    linear_offsets_in = collect_linear_offsets(view_shapes_in, strides_in, starts_in)
    linear_offset_out = collect_linear_offsets([view_shape_out], [stride_out], [start_out])[0]

    out_dim = len(squeeze(view_shape_out))
    
    #if some input dimensions are collapsed, we need to shuffle around the strides
    strides_in = rebuild_strides(strides_in, view_shapes_in, out_dim)
    stride_out = rebuild_strides([stride_out], [view_shape_out], out_dim)[0]
    if debug: print(f'Strides in after rebuild: {strides_in} out: {stride_out}')
     
    #calculate the offsets. these are relative to output offsets, as we will just loop directly on the output indeces that are changed.
    #again, offsets on collapsed dimensions should be disregareded, as we are accounting for this trough the linear offset!
    
    offsets = build_offsets(start_out, starts_in, view_shapes_in, out_dim)
    if debug: print(f'Offsets after rebuilding: {offsets}')
    
    out_indeces = itertools.product(*[range(off, off + sh) for sh, off in zip(squeeze(view_shape_out), start_out)])
    
    # for ind in out_indeces:
        # lin_index_inputs = [get_lin_index(ind, stride_in, offset, linear_offset) for stride_in, offset, linear_offset in zip(strides_in, offsets, linear_offsets)]
        # lin_index_output = get_lin_index(ind, stride_out, [0]*out_dim, 0)
        # print(f'{ind}. out: {lin_index_output} ' + ' '.join([f'input_{i}: {lin_index}' for i, lin_index in enumerate(lin_index_inputs)]))

    print(f"""Negoiated strides...
          input strides: {strides_in} output_strides: {stride_out}
          input offsets: {offsets} input lin offset: {linear_offsets_in} output lin offest {linear_offset_out}""")

    return strides_in, stride_out, offsets, linear_offsets_in, linear_offset_out, out_indeces


def run_test(test_case, debug=True):
    #first, we build random data based on the shapes:
    inputs = [np.random.randint(0, 100, shape) for shape in test_case["shapes"][:-1]]
    starts_in, starts_out = test_case["starts"][:-1], test_case["starts"][-1]
    ends_in, ends_out = test_case["ends"][:-1], test_case["ends"][-1]
    
    input_slices = [tuple(slice(st, end if end !=0 else None) for st, end in zip(starts, ends)) for starts, ends in zip(starts_in, ends_in)]
    output_slice = tuple(slice(st, end if end !=0 else None) for st, end in zip(starts_out, ends_out))
    
    output = np.zeros(test_case["shapes"][-1])
    output_squeezed = np.squeeze(output[output_slice])
    output_squeezed = (inputs[0][input_slices[0]]) + (inputs[1][input_slices[1]])
    
    strides_in, stride_out, offsets, linear_offsets, linear_offset_out, out_indeces = negotiate_strides(test_case, debug=debug)
    
    A_strides, B_strides = strides_in
    A_offset, B_offset = offsets
    A_lin_offset, B_lin_offset = linear_offsets
    out_dim = len(stride_out)
    
    inputs_flat = [inp.flatten() for inp in inputs]
    output_flat = np.zeros_like(output.flatten())

    for ind in out_indeces:
        A_ind, B_ind = get_lin_index(ind, A_strides, A_offset, A_lin_offset), get_lin_index(ind, B_strides, B_offset, B_lin_offset)
        A_val = inputs_flat[0][A_ind]
        B_val = inputs_flat[1][B_ind]
        out_ind = get_lin_index(ind, stride_out, [0]*out_dim, linear_offset_out)
        if debug: print(f'{ind}: out_ind: {out_ind}. A_ind: {A_ind}, B_ind: {B_ind}. A_val: {A_val}, B_val: {B_val}')
        output_flat[out_ind] = A_val + B_val
     
    if debug:
        print(f'Homerolled sum: {output_flat.sum()}')
        print(f'Numpy sum: {output_squeezed.sum()}')
        # print(output_flat)
        # print(output_squeezed)
    assert(output_flat.sum() == output_squeezed.sum())    

test_cases = [
    # {"shapes": [(32, 32, 4, 3), (1,), (32, 32, 4,)],
    #   "starts": [(0, 0, 0, 0), (0,), (0, 0, 0)],
    #   "ends": [(0, 0, 0, -2), (0,), (0, 0, 0,)]},
    # {"shapes": [(4, 4, 4), (4,), (4, 4, 4,)],
    #   "starts": [(0, 0, 1), (1,), (0, 0, 1)],
    #   "ends": [(0, 0, -1), (-1,), (0, 0, -1,)]},
    # {"shapes": [(32, 32, 4, 3), (32, 32, 4), (28, 28, 4,)],
    #   "starts": [(2, 2, 0, 0), (2, 2, 0), (0, 0, 0)],
    #   "ends": [(-2, -2, 0, -2), (-2, -2, 0), (0, 0, 0,)]},
    # {"shapes": [(28, 28, 4,), (1,), (28, 28, 4,)],
    #   "starts": [(0, 0, 4-1,), (0,), (0, 0, 4-1)],
    #   "ends": [(0, 0, 0), (0,), (0, 0, 0,)]},
    #   {"shapes": [(28, 28, 4), (1,), (28, 28, 4,)],
    # "starts": [(0, 0, 4-1), (0,), (0, 0, 4-1)],
    # "ends": [(0, 0, 0,), (0, ), (0, 0, 0,)]},
      # {"shapes": [(3, 1), (1, 3), (3, 3,)],
      # "starts": [(0, 0,), (0, 0,), (0, 0,)],
      # "ends": [(0, 0,), (0, 0,), (0, 0, )]}
    {"shapes": [(6, 6,), (6, 1,), (6, 6,)],
    "starts": [(2, 2, ), (2, 0,), (2, 2,)],
    "ends": [(-2, -2, ), (-2, 0), (-2, -2,)]}


    ]

for test_case in test_cases:
    run_test(test_case, debug=True)