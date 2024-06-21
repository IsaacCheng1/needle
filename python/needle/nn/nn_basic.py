"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None

        # initialize W and b with kaiming uniform distribution
        self.weight = Parameter(
                    init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype, requires_grad=True)
        )

        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1, requires_grad=True).transpose())

    def forward(self, X: Tensor) -> Tensor:
        output = X @ self.weight
        if self.bias:
            return output + self.bias.broadcast_to(output.shape)


class Flatten(Module):
    def forward(self, X):
        B = X.shape[0]
        return X.reshape((B, -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        output = x
        for module in self.modules:
            output = module(output)
        return output

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # B stands for batch size
        B = logits.shape[0]

        # k stands for # of classes
        k = logits.shape[1]

        # one-hot encoding of Y
        one_hot_Y = init.one_hot(k, y, requires_grad=True) # (B,k)

        loss = ops.logsumexp(logits, axes=(1,)) - (one_hot_Y * logits).sum(axes=(1,))

        loss = loss.sum() / B

        return loss

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))

        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            m, n = x.shape

            # compute the mean over the _batch_ dimension
            x_sum = x.sum(axes=(0,))
            x_mean = x_sum / m # (1,n)

            # compute the running average of mean
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.data # (1, n)

            x_subtract_mean = x - x_mean.broadcast_to(x.shape) # (m,n)

            # compute the variance over the _batch_ dimension
            x_var = x_subtract_mean ** 2  # (m,n)
            x_var = x_var.sum(axes=(0,))  # (1,n)
            x_var = x_var / m # (1,n)

            # compute the running average of variance
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var.data # (1, n)

            x_var += self.eps # (1,n)
            x_var = x_var ** (1/2) # (1,n)

            batch_normalized_x = x_subtract_mean / x_var.broadcast_to(x.shape)
            y = batch_normalized_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

            return y
        else:
            running_mean = self.running_mean.broadcast_to(x.shape)
            running_var = self.running_var.broadcast_to(x.shape)

            batch_normalized_x =  (x - running_mean) / ((running_var + self.eps) ** (1 / 2))

            y = batch_normalized_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

            return y

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        m, n = x.shape

        # compute the empirical mean of x
        x_sum = x.sum(axes=(1,))
        x_mean = x_sum / n # (m,)
        x_mean = x_mean.reshape((m, 1))

        x_subtract_mean = x - x_mean.broadcast_to(x.shape) # (m,n)

        # compute the empirical variance of x
        x_var = x_subtract_mean ** 2 # (m,n)
        x_var = x_var.sum(axes=(1,)) # (m,)
        x_var = x_var / n # (m,)
        x_var += self.eps # (m,)
        x_var = x_var ** (1/2) # (m,)
        x_var = x_var.reshape((m, 1))

        normalized_x = x_subtract_mean / x_var.broadcast_to(x.shape)

        y = normalized_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

        return y


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # # draw sample from bernoulli distribute
            # # bernoulli is a special distribution of binomial
            # dropout = Tensor(np.random.binomial(n=1, p=1-self.p, size=x.shape)) # element is either 1 or 0

            dropout = init.randb(*x.shape, p=1-self.p, requires_grad=True)
            dropout = dropout * (1 / (1 - self.p))

            return dropout * x
        else:
            return Identity()(x)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
