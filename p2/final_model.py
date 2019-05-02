import numpy as np
import matplotlib.pyplot as plt


beta = 717
mu = 260
gamma = 1290
rho = 0.3
alpha = 1.28 * 10**-4


def h(t):
    return 64 / (1 + 212.33 * np.exp(-2.28*t))


def w(t):
    return 6980 / (1 + 697 * np.exp(-1.256*t))


def f(k):
    return gamma * w(k)


def c(k):
    return (beta + gamma) / (1 - rho) * w(k/12) + mu / (1 - rho) * h(k/12)


if __name__ == "__main__":
    first = c(200000)
    gamma = 0.1*gamma
    daily_init = c(200000)
    rho = 0.01 * rho
    daily_new = c(200000)
    print(first, first/30)
    print(daily_init, daily_init/30)
    print(daily_new, daily_new/30)
    print(daily_init / first)
    print(daily_new / first)
    print(daily_new / daily_init)
