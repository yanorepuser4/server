#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse

import numpy as np
import tensorrt as trt


def np_to_trt_dtype(np_dtype):
    if np_dtype == bool:
        return trt.bool
    elif np_dtype == np.int8:
        return trt.int8
    elif np_dtype == np.int32:
        return trt.int32
    elif np_dtype == np.uint8:
        return trt.uint8
    elif np_dtype == np.float16:
        return trt.float16
    elif np_dtype == np.float32:
        return trt.float32
    return None


def create_plan_shape_tensor_modelfile(
    models_dir, model_version, io_cnt, max_batch, dtype, shape, profile_max_size
):
    # Note that resize layer does not support int tensors.
    # The model takes two inputs (INPUT and DUMMY_INPUT)
    # and produce two outputs.
    # OUTPUT : The shape of resized output 'DUMMY_OUTPUT'.
    # DUMMY_OUTPUT : Obtained after resizing 'DUMMY_INPUT'
    # to shape specified in 'INPUT'.
    # Note that values of OUTPUT tensor must be identical
    # to INPUT values

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    if max_batch == 0:
        shape_with_batchsize = len(shape)
        dummy_shape = [-1] * shape_with_batchsize
    else:
        shape_with_batchsize = len(shape) + 1
        dummy_shape = [-1] * shape_with_batchsize

    trt_dtype = np_to_trt_dtype(dtype)
    trt_memory_format = trt.TensorFormat.LINEAR
    for io_num in range(io_cnt):
        in_node = network.add_input(
            "INPUT{}".format(io_num), trt.int32, [shape_with_batchsize]
        )
        in_node.allowed_formats = 1 << int(trt_memory_format)
        dummy_in_node = network.add_input(
            "DUMMY_INPUT{}".format(io_num), trt_dtype, dummy_shape
        )
        dummy_in_node.allowed_formats = 1 << int(trt_memory_format)
        resize_layer = network.add_resize(dummy_in_node)
        resize_layer.set_input(1, in_node)
        out_node = network.add_shape(resize_layer.get_output(0))

        dummy_out_node = resize_layer.get_output(0)
        out_node.get_output(0).name = "OUTPUT{}".format(io_num)

        dummy_out_node.name = "DUMMY_OUTPUT{}".format(io_num)

        dummy_out_node.dtype = trt_dtype
        network.mark_output(dummy_out_node)
        dummy_out_node.allowed_formats = 1 << int(trt_memory_format)

        out_node.get_output(0).dtype = trt.int32
        network.mark_output_for_shapes(out_node.get_output(0))
        out_node.get_output(0).allowed_formats = 1 << int(trt_memory_format)

        if trt_dtype == trt.int8:
            in_node.dynamic_range = (-128.0, 127.0)
            out_node.get_output(0).dynamic_range = (-128.0, 127.0)

    config = builder.create_builder_config()

    min_prefix = []
    opt_prefix = []
    max_prefix = []

    if max_batch != 0:
        min_prefix = [1]
        opt_prefix = [max(1, max_batch)]
        max_prefix = [max(1, max_batch)]

    min_shape = min_prefix + [1] * len(shape)
    opt_shape = opt_prefix + [8] * len(shape)
    max_shape = max_prefix + [profile_max_size] * len(shape)

    profile = builder.create_optimization_profile()
    for io_num in range(io_cnt):
        profile.set_shape_input(
            "INPUT{}".format(io_num), min_shape, opt_shape, max_shape
        )
        profile.set_shape(
            "DUMMY_INPUT{}".format(io_num), min_shape, opt_shape, max_shape
        )

    config.add_optimization_profile(profile)

    flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
    datatype_set = set([trt_dtype])
    for dt in datatype_set:
        if dt == trt.int8:
            flags |= 1 << int(trt.BuilderFlag.INT8)
        elif dt == trt.float16:
            flags |= 1 << int(trt.BuilderFlag.FP16)
    config.flags = flags

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    with open(models_dir + "/model.plan", "wb") as f:
        f.write(engine_bytes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Top-level model directory"
    )
    FLAGS, unparsed = parser.parse_known_args()

    create_plan_shape_tensor_modelfile(
        models_dir=FLAGS.models_dir, model_version=1, io_cnt=1, max_batch=8,
        dtype=np.float32, shape=[-1, -1], profile_max_size=32
    )
