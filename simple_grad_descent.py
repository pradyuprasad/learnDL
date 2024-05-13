import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

a, b, c, = 0, 0, 0

def make_quadratic(a:int, b:int, c:int) -> callable:
    return lambda x:  a*x**2 + b*x + c

f: callable = make_quadratic(3, -1, 2)

def loss(guess: callable, actual:callable) -> float:
        x_values:np.ndarray = np.arange(-10, 10, 0.1)
        predicted = guess(x_values)
        actual_values = actual(x_values)
        return np.sum(((predicted - actual_values) ** 2)) / len(x_values)

learn_rate = 0.0001
prev_loss = float('inf')
for i in range(1000001):
    guess = make_quadratic(a, b, c)
    a_grad = (loss(make_quadratic(a+0.01, b, c), f) - loss(guess, f)) / 0.01 
    b_grad = (loss(make_quadratic(a, b+0.01, c), f) - loss(guess, f)) / 0.01 
    c_grad = (loss(make_quadratic(a, b, c+0.01), f) - loss(guess, f)) / 0.01
    a -= a_grad*learn_rate
    b -= b_grad * learn_rate
    c -= c_grad * learn_rate
    curr_loss = loss(guess, f)
    if curr_loss > prev_loss or curr_loss < 0.01:
        break
    if i % 10000 == 0:
        print(f'Iteration {i}, Loss: {curr_loss}')
    prev_loss = loss(guess, f)

    
print(loss(make_quadratic(a, b, c), f))
print(a, b, c)
print(round(a), round(b), round(c))