import numpy as np


class OdeSolver:
    def __init__(self, t0=0, y0=[], y=[]):
        self.t0 = np.float64(t0)
        self.y0 = np.array(y0, dtype=np.float64)
        self.y = y

        self.tn = None
        self.yn = None
        self._h = None
        self._tf = None
        self._steps = None

        self._k1 = None
        self._k2 = None
        self._k3 = None
        self._k4 = None

    def run(self, tf, steps, t0=None, y0=None, y=None):
        if t0 is not None:
            self.t0 = t0
        if y0 is not None:
            self.y0 = y0
        if y is not None:
            self.y = y

        self._tf = tf
        self._steps = steps
        self._h = (tf-self.t0)/steps

        self.tn = [self.t0]
        self.yn = [self.y0]

        for _ in range(steps):
            tnext, ynext = self._step()
            self.tn.append(tnext)
            self.yn.append(ynext)

    def plot(self, labels=None):
        import matplotlib.pyplot as plt
        dims = len(self.y0)
        if labels is None:
            lables = ['Dim {}'.format(i+1) for i in range(dims)]

        for i in range(dims):
            plt.subplot(dims, 1, i+1)
            plt.plot(self.tn, np.array(self.yn)[:, i], '-o')
            plt.xlabel('Time')
            plt.ylabel(labels[i])
        plt.show()

    def _step(self):
        self._setK1()
        self._setK2()
        self._setK3()
        self._setK4()
        sum = self._k1 + 2*self._k2 + 2*self._k3 + self._k4
        ynext = self.yn[-1] + sum/6
        tnext = self.tn[-1] + self._h
        return tnext, ynext

    def _setK1(self):
        ret = []
        h = self._h
        tn = self.tn[-1]
        yn = self.yn[-1]
        for f in self.y:
            ret.append(h*f(tn, yn))
        self._k1 = np.array(ret)

    def _setK2(self):
        ret = []
        h = self._h
        tn = self.tn[-1]
        yn = self.yn[-1]
        k1 = self._k1
        for f in self.y:
            ret.append(h*f(tn+h/2, yn+k1/2))
        self._k2 = np.array(ret)

    def _setK3(self):
        ret = []
        h = self._h
        tn = self.tn[-1]
        yn = self.yn[-1]
        k2 = self._k2
        for f in self.y:
            ret.append(h*f(tn+h/2, yn+k2/2))
        self._k3 = np.array(ret)

    def _setK4(self):
        ret = []
        h = self._h
        tn = self.tn[-1]
        yn = self.yn[-1]
        k3 = self._k3
        for f in self.y:
            ret.append(h*f(tn+h, yn+k3))
        self._k4 = np.array(ret)
