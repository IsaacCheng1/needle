"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            grad = w.grad.realize_cached_data()

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * w.realize_cached_data()

            if self.momentum > 0:
                # keep track of a moving average of multiple previous gradients for grad w.

                # init as 0
                if w not in self.u:
                    self.u[w] = ndl.zeros(*w.shape, requires_grad=False).realize_cached_data()

                self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * grad

                grad = self.u[w]

            # 1st half is for L2 regularization
            # 2nd half is vanilla SGD
            w.data = w.data - self.lr * grad

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for w in self.params:
            grad = w.grad.realize_cached_data()

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * w.realize_cached_data()

            # keep track of a moving average of multiple previous gradients and square of gradients for grad w.
            # init as 0
            if w not in self.m:
                self.m[w] = ndl.zeros(*w.shape, requires_grad=False).realize_cached_data()
                self.v[w] = ndl.zeros(*w.shape, requires_grad=False).realize_cached_data()

            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)

            # make bias correction
            unbiased_m = self.m[w] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[w] / (1 - self.beta2 ** self.t)

            grad = unbiased_m / (unbiased_v ** 0.5 + self.eps)

            w.data = w.data - self.lr * grad