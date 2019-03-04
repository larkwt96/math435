from echonn.sys import DynamicalSystem, SystemSolver
import matplotlib.pyplot as plt
import numpy as np

class TropicLevels(DynamicalSystem):
    def __init__(self, rate, capacity, efficiency, outside_drain, levels=3):
        super().__init__(dim=levels)
        self.r = rate # 1
        self.k = np.ones(levels)
        self.k[0] *= capacity
        self.e = np.array(efficiency) # levels - 1
        for i in range(1, len(self.k)):
            self.k[i] *= self.e[i-1] * self.k[i-1]
        self.drain = outside_drain

    def fun(self, t, v):
        r = self.r
        k = self.k
        e = self.e
        y = np.zeros_like(v)

        trans = r * v*v/k
        y[0] = r * v[0]
        y[1:] = e*trans[:-1]
        y -= trans
        return y - self.drain(t, v)

if __name__ == '__main__':
    def drain(t, v):
        if 13 < t and t < 15:
            return np.array([0, 10, 0])
        else:
            return np.zeros(3)
    def idem(t, v):
        return np.zeros(3)
    tropic_problem = TropicLevels(1, 1000, [.1, .12], drain)
    solver = SystemSolver(tropic_problem)
    solver.run([0, 20], [1, 0, 0])
    solver.plotnd()
    plt.show(True)

