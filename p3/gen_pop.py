#!/usr/bin/env python3

from person import Person
import matplotlib.pyplot as plt
from state import State
import os
import pickle


if __name__ == "__main__":
    load = False
    fn = 'data.p'
    if os.path.exists(fn) and load:
        state = pickle.load(open(fn, 'rb'))
    else:
        state = State()
        pickle.dump(state, open(fn, 'wb'))
    state.plot_people()
    state.plot_blocks()
    state.plot_district_even()
    plt.show(True)
