#!/usr/bin/env python3

import numpy as np
import time
from person import Person
import matplotlib.pyplot as plt
from state import State
import os
import pickle


def load_state(load=False, fn='data.p'):
    if os.path.exists(fn) and load:
        state = pickle.load(open(fn, 'rb'))
    else:
        state = State()
        pickle.dump(state, open(fn, 'wb'))
    return state


def collect_dist():
    state = load_state()
    state.plot_people()
    state.plot_blocks()

    #
    load = True
    fn = 'distr.p'
    if os.path.exists(fn) and load:
        districts = pickle.load(open(fn, 'rb'))
    else:
        districts = []
        iters = 500  # minute per hundred
        appx = None
        start = time.time()
        for i in range(iters):
            update_i = max(1, iters//10)
            if i % update_i == 0:
                if appx is None:
                    appx = ''
                else:
                    dur = time.time() - start
                    total = iters * dur / i
                    rem = total - dur
                    unit = 's'
                    if rem > 60:
                        rem = f'{rem / 60:60.1}'.strip()
                        unit = 'm'
                    appx = f' ({int(rem)} {unit} remaining)'
                print(f'Iteration {i}{appx}')

            districts.append(state.district_even())
        pickle.dump(districts, open(fn, 'wb'))
    for district in districts[:3]:
        state.plot_district_even(district=district)
    compacts = []
    for district in districts:
        compactness = state.calc_compactness(district)
        compacts.append(compactness)
    distribution = [compact[0] for compact in compacts]

    # graphing
    plt.figure()
    plt.hist(distribution, bins=30)
    plt.xlabel('Compactness')
    plt.ylabel('Count')
    plt.title('Distribution of compactness')


def compact_vs_blocks():
    ns = list(range(5, 20))
    compacts = []
    for n in ns:
        state = State(block_width=n)
        districts = state.district_even()
        compactness = state.calc_compactness(districts)
        compacts.append(compactness)
        # print(compactness[0])
    ns = [n*n for n in ns]
    plt.figure()
    plt.plot(ns, [compact[0] for compact in compacts])
    plt.ylim(bottom=0)
    plt.xlabel('Number of Blocks')
    plt.ylabel('Compactness')
    plt.title('Variance in compactness decreases with number of blocks')


def compact_vs_seats():
    state = State()
    ns = list(range(2, 10))
    compacts = []
    for n in ns:
        districts = state.district_even(n=n)
        compactness = state.calc_compactness(districts)
        compacts.append(compactness)
        # print(compactness[0])
    plt.figure()
    plt.plot(ns, [compact[0] for compact in compacts])
    plt.ylim(bottom=0)
    plt.xlabel('Number of Seats')
    plt.ylabel('Compactness')
    plt.title('Compactness decreases with number of seats')


if __name__ == "__main__":
    print('distribution...')
    collect_dist()
    print('compact vs seats...')
    # compact_vs_seats()
    print('compact vs blocks...')
    # compact_vs_blocks()
    plt.show()
