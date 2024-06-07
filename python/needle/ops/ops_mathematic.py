"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        return (out_grad * self.scalar)*(node.inputs[0]**(self.scalar - 1))

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad * (rhs ** -1), out_grad * -lhs * (rhs ** -2)

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        return out_grad * (1 / self.scalar)

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        # axes is either None or an order of 2 axis (axis1, axis2)
        self.axes = axes

    def compute(self, a):
        num_axes = len(a.shape)
        array_api_axes = list(range(num_axes))

        if self.axes == None:
            array_api_axes[-1], array_api_axes[-2] = array_api_axes[-2], array_api_axes[-1]
        else:
            axis_1 = self.axes[0]
            axis_2 = self.axes[1]
            array_api_axes[axis_1], array_api_axes[axis_2] = array_api_axes[axis_2], array_api_axes[axis_1]
        return array_api.transpose(a, axes=array_api_axes)

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        return out_grad.reshape(node.inputs[0].shape)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(ori_shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lhs_partial_adjoint = out_grad.matmul(rhs.transpose())
        rhs_partial_adjoint = lhs.transpose().matmul(out_grad)

        if lhs.shape != lhs_partial_adjoint.shape:
            # sum along the "stack" axes, along which matrix are stacked
            num_stack_axes = len(lhs_partial_adjoint.shape) - len(lhs.shape)
            lhs_partial_adjoint = lhs_partial_adjoint.sum(axes=tuple(range(num_stack_axes)))

        if rhs.shape != rhs_partial_adjoint.shape:
            # sum along the "stack" axes, along which matrix are stacked
            num_stack_axes = len(rhs_partial_adjoint.shape) - len(rhs.shape)
            rhs_partial_adjoint = rhs_partial_adjoint.sum(axes=tuple(range(num_stack_axes)))

        assert lhs.shape == lhs_partial_adjoint.shape
        assert rhs.shape == rhs_partial_adjoint.shape

        return lhs_partial_adjoint, rhs_partial_adjoint

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        return out_grad * (node.inputs[0] ** -1)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        a_copy = array_api.copy(a)
        a_copy[a_copy < 0] = 0
        return a_copy

    def gradient(self, out_grad, node):
        assert len(node.inputs) == 1
        gradient = array_api.copy(node.realize_cached_data())
        gradient[gradient < 0] = 0
        gradient[gradient > 0] = 1
        return out_grad * Tensor(gradient)

def relu(a):
    return ReLU()(a)
