from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        maximum_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        subtracted_Z = Z - maximum_Z
        exp_Z = array_api.exp(subtracted_Z)
        sum_Z = array_api.sum(exp_Z, axis=self.axes, keepdims=True)
        log_Z = array_api.log(sum_Z)
        res = log_Z + maximum_Z
        return array_api.squeeze(res)

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        Z = node.inputs[0].realize_cached_data()
        maximum_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_Z = exp(Tensor(Z - maximum_Z)) # exp_Z is a Tensor

        # the dimension of Tensor sum_Z is different from Tensor exp_Z.
        # but it is the same as the dimension of Tensor out_grad and Tensor node.
        sum_Z = summation(exp_Z, axes=self.axes)
        intermediate_res = out_grad / sum_Z # this is the intermediate res in order to compute the final partial adjoint

        # play with shape and axes
        expand_shape = list(Z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1

        # reshape
        partial_adj = intermediate_res.reshape(expand_shape).broadcast_to(Z.shape)

        return partial_adj * exp_Z

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

