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
        self.tensor = torch.cuda.sparse.FloatTensor(0, 0)

class RandomKCompressor(Compressor):
    ### update the hash for each layer using layer name
    # torch.cuda.manual_seed_all(hash(name) + global_step)
    ### use random vector generated from a uniform distribution
    # mask = torch.cuda.FloatTensor(flatten_grad.shape).uniform_(0, 1).ge(1-topk)
    @staticmethod
    def compress(tensor, ratio):
        ret = stru()
        ret.size = tensor.shape
        flatten_grad = tensor.resize(-1)
        torch.backends.cudnn.enabled = False
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        compress_grad = flatten_grad.clone()
        mask = torch.randperm(flatten_grad.numel(), device=torch.device('cuda')).lt(flatten_grad.numel() * (1.0-ratio))
        compress_grad[mask] = 0
        compress_grad = compress_grad.to_sparse()
        ret.flag = True
        ret.tensor = compress_grad
        compress_grad = compress_grad.values()
        return compress_grad, ret

    @staticmethod
    def decompress(tensor, ctx):
        if ctx.flag == True:
            tensor_decompressed = ctx.tensor
            tensor_decompressed.values = tensor
            ctx.flag = False
        return tensor_decompressed.to_dense().reshape(ctx.size)

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
