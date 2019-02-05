import time
from ode_solver import OdeSolver


class IR:
    def __init__(self, gamma=1, y0=[10, 1], tf=24):
        def s(t, y):
            return -gamma*y[0]

        def i(t, y):
            return gamma*y[0]
        self.tf = tf
        self.model = OdeSolver(y0=y0, y=[s, i])

    def run(self, steps=None):
        if steps is None:
            steps = int(self.tf*2000)
        self.model.run(self.tf, steps)

    def plot(self, labels=["Susceptible", "Infected"]):
        self.model.plot(labels=labels)


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
            steps = int(self.tf*2000)
        self.model.run(self.tf, steps)

    def plot(self, labels=["Susceptible", "Infected"]):
        self.model.plot(labels=labels)

class SIR:
    def __init__(self, gamma1=1, gamma2=1, y0=[10, 1, 0], tf=24):
        def s(t, y):
            return - gamma1*y[0]*y[1]

        def i(t, y):
            return gamma1*y[0]*y[1] - gamma2*y[1]

        def r(t, y):
            return gamma2*y[1]
        
        self.tf = tf
        self.model = OdeSolver(y0=y0, y=[s, i, r])

    def run(self, steps=None):
        if steps is None:
            steps = int(self.tf*2000)
        self.model.run(self.tf, steps)

    def plot(self, labels=["Susceptible", "Infected", "Recovered"]):
        self.model.plot(labels=labels)

def find_gamma(pop=33413, initInf=4, infectionchance=0.75, dailypeople=42):
    chance = infectionchance
    gamma = dailypeople/24*chance/initInf/(pop-initInf)
    target = dailypeople*infectionchance+initInf
    eps = 1.0
    print('Finding best gamma1...')
    for i in range(1000):
        eps *= .8
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
    print('gamma:', gamma)
    return gamma

def runsir(pop=33413, initInf=4, infectionchance=0.75, dailypeople=42, gamma=None, gamma2=0.1, tf=200):
    if gamma is None:
        gamma = find_gamma(pop, initInf, infectionchance, dailypeople)
    gamma1 = gamma
    gamma2 = gamma2
    print('Running model...')
    model = SIR(gamma1=gamma1, gamma2=gamma2, y0=[pop-initInf, initInf, 0], tf=tf)
    model.run()
    model.plot()


def runsi(pop=33413, initInf=4, infectionchance=0.75, dailypeople=42, gamma=None):
    if gamma is None:
        gamma = find_gamma(pop, initInf, infectionchance, dailypeople)
    print('Running model...')
    model = SI(gamma=gamma, y0=[pop-initInf, initInf], tf=200)
    model.run()
    #model.plot()


if __name__ == "__main__":
    #runsi()
    runsir(gamma=2.726045493524211e-06, gamma2=1)
    runsir(gamma=2.726045493524211e-06, gamma2=.1)
    runsir(gamma=2.726045493524211e-06, gamma2=.01)
    runsir(gamma=2.726045493524211e-06, gamma2=.05, tf=500)

    #runsi(infectionchance=0)
    #runsi(infectionchance=.25)
    #runsi(infectionchance=.5)
    #runsi(infectionchance=.75)
    #runsi(infectionchance=1)
