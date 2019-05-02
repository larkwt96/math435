from echonn.sys import DynamicalSystem, SystemSolver
import matplotlib.pyplot as plt
import numpy as np

class EcoSystem(DynamicalSystem):
    def __init__(self, biome_args, num_biomes=None):
        super().__init__()
        biome_args = self.expand_biomes(biome_args, num_biomes)
        self.dim = self.get_levels(biome_args)
        self.biomes = [TropicSystem(**biome_arg) for biome_arg in biome_args]

    def get_levels(self, biome_args):
        total_levels = 0
        for biome_arg in biome_args:
            if 'levels' not in biome_arg:
                biome_arg['levels'] = 3
            total_levels += biome_arg['levels']
        return total_levels

    def expand_biomes(self, biome_args, num_biomes):
        if isinstance(biome_args, dict):
            if num_biomes is None:
                raise Exception('number of biomes not specified')
            else:
                biome_args = [biome_args for _ in range(num_biomes)]
        return biome_args

    def dragon_influence(self, t, v ,y):
        return y

    def fun(self, t, v):
        y = np.zeros_like(v)
        last_i = 0
        for biome in self.biomes:
            new_i = last_i+biome.levels
            y[last_i:new_i] += biome.fun(t, v[last_i:new_i])
            last_i = new_i
        y = self.dragon_influence(t, v, y)
        return y
        

class TropicSystem(DynamicalSystem):
    def __init__(self, rate, capacity, efficiency, levels=3):
        super().__init__(dim=levels)
        self.r = rate # 1
        self.k = np.ones(levels)
        self.k[0] *= capacity
        try:
            self.e = np.array([e for e in efficiency]) # levels - 1
        except TypeError:
            self.e = efficiency * np.ones(levels - 1)
        for i in range(1, len(self.k)):
            self.k[i] *= self.e[i-1] * self.k[i-1]
        self.levels = levels

    def fun(self, t, v):
        r = self.r
        k = self.k
        e = self.e

        trans = r * v*v/k
        y = np.zeros_like(v)
        y[0] += r * v[0]
        y[1:] += e*trans[:-1]
        y -= trans
        return y

if __name__ == '__main__':
    biome1 = {'rate': 1, 'capacity': 1000, 'efficiency': [.1, .12], 'levels':3}
    biome2 = {'rate': .1, 'capacity': 100, 'efficiency': [.1, .9], 'levels':3}
    biome3 = {'rate': .01, 'capacity': 5000, 'efficiency': [.1, .2], 'levels':3}
    problem = EcoSystem([biome1, biome2, biome3])
    solver = SystemSolver(problem)
    y0 = np.zeros(problem.dim)
    y0[0] = 1
    y0[3] = 1
    y0[4] = 1000
    y0[6] = 1
    solver.run([0, 200], y0)
    solver.plotnd()
    plt.show(True)

