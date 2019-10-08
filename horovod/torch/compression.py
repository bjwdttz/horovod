# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch
import numpy as np
import torch.sparse
import os 
import random
import math

class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor, ratio):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor, ratio):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor

class stru:
    def __init__(self):
        self.flag = False
        self.size = None
        self.mask = None
        self.tensor = None

def seed_torch(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RandomKCompressor(Compressor):
    ### use random vector generated from a uniform distribution
    # mask = torch.cuda.FloatTensor(flatten_grad.shape).uniform_(0, 1).ge(1-topk)
    @staticmethod
    def compress(tensor, ratio):
        #seed_torch()
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        ret = stru()
        ret.size = tensor.shape
        flatten_grad = tensor.reshape(-1)
        #print("ori", flatten_grad.shape)
        compress_grad = flatten_grad.clone()
        ret.mask = torch.randperm(flatten_grad.numel(), device=torch.device('cuda')).lt(int(math.ceil(flatten_grad.numel() * ratio)))
        compress_grad = compress_grad[ret.mask]
        ret.flag = True
        ret.tensor = flatten_grad
        #print("comp", compress_grad.shape)
        return compress_grad, ret

    @staticmethod
    def decompress(tensor, ctx):
        if ctx.flag == True:
            tensor_decompressed = ctx.tensor
            #print(tensor_decompressed.shape, ctx.mask.shape, tensor.shape)
            tensor_decompressed[ctx.mask] = tensor
            ctx.flag = False
        else:
            print("flag should be true!")
        #print(tensor_decompressed.shape)
        return tensor_decompressed.reshape(ctx.size)

class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor, ratio):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor

    randk = RandomKCompressor
