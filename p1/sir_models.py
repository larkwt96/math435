import time
from ode_solver import OdeSolver


class SI:
    def __init__(self, gamma=1, y0=[10, 1], tf=24):
        def s(t, y):
            return -gamma*y[0]*y[1]

        def i(t, y):
            return gamma*y[0]*y[1]
        self.tf = tf
        self.model = OdeSolver(y0=y0, y=[s, i])

    def run(self, steps=None):
        if steps is None:
            steps = int(self.tf*1000)
        self.model.run(self.tf, steps)

    def plot(self, labels=["Susceptible", "Infected"]):
        self.model.plot(labels=labels)


def runsi(pop=33413, initInf=4, infectionchance=0.75, dailypeople=42):
    chance = infectionchance
    gamma = dailypeople/24*chance/initInf/(pop-initInf)
    target = dailypeople*infectionchance
    eps = 1.0
    print('Finding best epsilon...')
    for i in range(1000):
        eps *= .9
        model = SI(gamma=gamma, y0=[pop-initInf, initInf], tf=24)
        model.run()
        value = model.model.yn[-1][1]
        print('err: {}:{}'.format(i, abs(value-target)))
        if abs(value-target) < 0.1:
            break
        elif value > target:
            gamma *= 1 - eps
        elif value < target:
            gamma *= 1 + eps
    print('Running model...')
    model = SI(gamma=gamma, y0=[pop-initInf, initInf], tf=200)
    model.run()
    model.plot()


if __name__ == "__main__":
    runsi()
