import numpy as np
from typing import List
import math

#inspired from Anderj Karpathy's micrograd (https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb)

class Value:
    data:float = None
    def __init__(self, v:int, _children=(), _op = '', label = ''):
        self.data: int = v
        self._prev: set = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
        self.grad = 0.0 # by default

    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(np.tanh(self.data), (self, ), 'tanh')
        def _backward():
            self.grad += (1 - np.tanh(self.data) ** 2) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
          self.grad += out.data * out.grad 
        out._backward = _backward

        return out

    
    def backward(self):
        # backward is for external facing API
        # _backward is a lambda that stores the function to find the gradient of the node
        topo = []
        visited = set()
        def topo_sort(node):
            if node not in visited: # post order DFS
                visited.add(node)
                for v in node._prev:
                    topo_sort(v)
                topo.append(node)
        
        topo_sort(self)
        topo = reversed(topo) # because we start our toposort from the end with edges reversed
        self.grad = 1.0
        for node in topo:
            node._backward() # find the gradient of each of the nodes in topological order
    
    def __radd__(self, other): # other + self
        return self + other
    def __rmul__(self, other): # other * self
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
            out._backward = _backward
    
        return out


