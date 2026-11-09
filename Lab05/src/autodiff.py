"""A small scalar reverse-mode automatic differentiation engine."""

from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    """A scalar value that records its computation graph and accumulates gradients."""

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = _children
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @staticmethod
    def _value(other: Value | float) -> Value:
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: Value | float) -> Value:
        other = self._value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + -self._value(other)

    def __rsub__(self, other: Value | float) -> Value:
        return self._value(other) - self

    def __mul__(self, other: Value | float) -> Value:
        other = self._value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __pow__(self, exponent: float) -> Value:
        exponent = float(exponent)
        out = Value(self.data**exponent, (self,), f"**{exponent:g}")

        def backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1.0) * out.grad

        out._backward = backward
        return out

    def __truediv__(self, other: Value | float) -> Value:
        other = self._value(other)
        return self * (other**-1.0)

    def __rtruediv__(self, other: Value | float) -> Value:
        return self._value(other) / self

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def backward() -> None:
            self.grad += (out.data > 0.0) * out.grad

        out._backward = backward
        return out

    def sigmoid(self) -> Value:
        if self.data >= 0.0:
            value = 1.0 / (1.0 + math.exp(-self.data))
        else:
            exp_value = math.exp(self.data)
            value = exp_value / (1.0 + exp_value)
        out = Value(value, (self,), "sigmoid")

        def backward() -> None:
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = backward
        return out

    def sum(self) -> Value:
        """Return a scalar sum (identity for this scalar-valued engine)."""
        out = Value(self.data, (self,), "sum")

        def backward() -> None:
            self.grad += out.grad

        out._backward = backward
        return out

    def backward(self) -> None:
        """Backpropagate from this node, accumulating gradients on every visit."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topology(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topology(parent)
                topo.append(node)

        build_topology(self)
        accumulated = {node: node.grad for node in topo}
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        for node in topo:
            node.grad += accumulated[node]

    def __repr__(self) -> str:
        return f"Value(data={self.data:g}, grad={self.grad:g})"
