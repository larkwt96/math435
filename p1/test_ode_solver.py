import math
import unittest
import ode_solver as odes


class TestModel(unittest.TestCase):
    def setUp(self):
        def y0(t, y):
            res = t*(y[0]+y[1])
            return res

        def y1(t, y):
            return t*(y[0]+2*y[1])

        def y2(t, y):
            return y[0]

        self.odes = odes.OdeSolver(0, [1, 1], y=[y0, y1])
        self.ode = odes.OdeSolver(0, [1], [y2])

    def test_bool(self):
        self.ode.run(5, 10000)
        self.odes.run(5, 100)
        self.assertAlmostEqual(math.exp(5), self.ode.yn[-1][0])
