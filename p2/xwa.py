#!/usr/bin/env python3

from echonn.sys import DynamicalSystem, SystemSolver
import matplotlib.pyplot as plt
import numpy as np

class XwaModel(DynamicalSystem):
    Grass = 10000
    Arctic = 2000
    Arid = 7000

    def __init__(self, flight_rate=50, fire_rate=80, biome=None):
        super().__init__(dim=2)
        if biome is None:
            biome = XwaModel.Grass
        self.flight_rate = flight_rate
        self.fire_rate = fire_rate
        self.biome_carry_capacity = biome
        self.biome_growth_rate = 10

    def fun(self, t, v):
        w,r = v
        x = r*w
        a = self.flight_rate + self.fire_rate
        dw = (x - w * a) / 10**4
        dr = -x + self.biome_growth_rate*r - self.biome_growth_rate*r**2/self.biome_carry_capacity
        if t < 100:
            dw = 0
        return dw, dr

if __name__ == '__main__':
    problem = XwaModel()
    solver = SystemSolver(problem)
    y0 = np.array([8, 5000])
    solver.run([0, 250], y0)
    solver.plotnd()
    plt.show(True)
