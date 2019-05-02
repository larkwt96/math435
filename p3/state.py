import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import product
import numpy as np
from sklearn.cluster import KMeans
from random import shuffle

# local imports
from block import Block
from person import Person
from kmeans import ManKMeans


"""
The state is a square, and there are distributions which the properties of
users follow. Properties include voting preference and possible race or
religion qualities.

Need to generate people in a cartesian plane. They will be focus around points, normally. The points will have different sizes.


Generate cities and sizes.
Generate people around cities and sizes.
Group people in blocks.
Summarize qualities for blocks.
Group blocks for district.
Render grouping.
"""


class State:
    """
    People live in a [0, 1)x[0, 1) block (which algorithms depend on so don't
    change)
    """

    def __init__(self, num_people=None, block_width=15):
        self.people_scale = 1000
        if num_people is None:
            num_people = int(5.696 * 10**6 / self.people_scale)
        self.num_people = int(num_people)
        self.block_width = block_width
        self.people = {}
        self.blocks = [[Block() for _ in range(block_width)]
                       for _ in range(block_width)]
        self.gen_pop()
        self.gen_blocks()

    def get_block_pos(self, pos):
        pos_scaled = self.block_width * np.array(pos)
        return tuple(pos_scaled.astype(int))

    def get_block(self, pos):
        blockx, blocky = self.get_block_pos(pos)
        return self.get_block_xy(blockx, blocky)

    def get_block_xy(self, x, y):
        return self.blocks[x][y]

    def gen_blocks(self):
        for pos in self.people:
            block = self.get_block(pos)
            block.add_person(self.people[pos])

    def gen_pop(self):
        for _ in range(self.num_people):
            pos = tuple(np.random.rand(2))
            self.people[pos] = Person()

    def get_block_voting(self):
        z = self.get_proportion_z()
        z[z < .5] = 0
        z[z == .5] = np.random.randint(2)
        z[z > .5] = 1
        return z.astype(int)

    def plot_district(self, district):
        classes, n, *rest = district
        plt.figure()
        plt.imshow(classes/n, cmap='viridis', origin='lower')

        # draw gridlines
        plt.grid(which='major', axis='both',
                 linestyle='-', color='k', linewidth=1)
        tick_locs = np.arange(self.block_width) - .5
        plt.xticks(tick_locs, range(self.block_width), ha='left')
        plt.yticks(tick_locs, range(self.block_width), va='bottom')

    def plot_district_even(self, n=None, district=None):
        if n is None:
            n = self.get_seats()
        if district is None:
            district = self.district_even(n)
        self.plot_district(district)
        centers = self.block_width*district[2]-.5
        plt.scatter(centers[:, 0], centers[:, 1], c='k', s=5)

    def calc_compactness(self, district):
        classes, n, centers = district
        block_locs = self.get_block_centers()
        scores = []
        for classi in range(n):
            classi_locs = np.where(classes == classi)
            sq_dist = (block_locs[classi_locs] - centers[classi, :])**2
            sq_dist_sum = np.sum(sq_dist, axis=1)
            score = np.mean(sq_dist_sum, axis=0)
            scores.append(score)
        return np.mean(scores), scores

    def get_block_center(self, x, y):
        """ returns position of the xth block and yth block, origin lower """
        block_len = 1 / self.block_width
        locs = np.arange(0, 1, block_len) + block_len/2
        return locs[x], locs[y]

    def get_block_centers(self):
        """Returns blocks in a (rows, cols, 2) matrix."""
        block_len = 1 / self.block_width
        locs = np.arange(0, 1, block_len) + block_len/2
        block_centers = np.array([[[x, y] for x in locs] for y in locs])
        return block_centers

    def get_seats(self):
        us_pop = 327.2 * 10**6
        seats = 435
        people_per_seat = us_pop / seats / self.people_scale
        # is 1, but we want interesting results, since 1 can't be gerrymandered
        res = max(int(self.num_people / people_per_seat), 2)
        return res

    def objective_func(self):
        """ how do we auto generate equal districts?

        we can use kmeans to generate an initial partition. Then, we need to
        make sure criteria are met. The criteria is the population, but we
        want to minimize compactness. What's the best approach?
        """
        pass

    def calc_class_pop(self, classes, n):
        """
        returns two arrays. pops and counts. It is the populations and
        block counts for each district. class is index.
        """
        pops = np.zeros(n)
        class_count = np.zeros(n)
        for x in range(self.block_width):
            for y in range(self.block_width):
                # pref is the district is belongs to
                pref = classes[y, x]
                block = self.get_block_xy(x, y)
                pop_count = len(block.people)
                pops[pref] += pop_count
                class_count[pref] += 1
        return pops, class_count

    def get_kmeans_classes(self, n=None, manual=False):
        """
        returns classes, num classes (n) and k means centers (centers), row
        by col of class
        """
        if n is None:
            n = self.get_seats()
        X = np.array([[x, y] for x, y in self.people.keys()])
        if manual:
            kmeans = ManKMeans(n_clusters=n, random_state=0).fit(X)
        else:
            kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        block_centers = self.get_block_centers()
        old_shape = block_centers.shape
        classes = kmeans.predict(block_centers.reshape(-1, 2))
        classes = classes.reshape(*old_shape[:-1])
        return classes, n, centers

    def get_dist_center(self, blocks):
        block_poses = [self.get_block_center(x, y) for y, x in blocks]
        return np.mean(block_poses, axis=0)

    def get_dist_centers(self, classes, n):
        """ returns array of centers for each class (relative to center of
        block). shape: class, dim
        """
        blocks = [self.get_blocks(classes, classi) for classi in range(n)]
        return np.array([self.get_dist_center(block_group)
                         for block_group in blocks])

    def get_blocks(self, classes, classi):
        """ returns blocks in y, x coords """
        top_blocks = np.where(classes == classi)
        return list(zip(*top_blocks))

    def dist(self, u, v):
        # squared distance since actual dist isnt needed
        return np.sum((u-v)**2)

    def get_area(self, classes, blocki):
        ''' blocki is y, x and returns iterable '''
        y, x = blocki
        n = classes.shape[0]
        minx = max(0, x-1)
        maxx = min(x+2, n)
        miny = max(0, y-1)
        maxy = min(y+2, n)
        xs = list(range(minx, maxx))
        ys = list(range(miny, maxy))
        return product(ys, xs)

    def get_area_neighbors(self, classes, blocki):
        y, x = blocki
        n = classes.shape[0]
        minx = max(0, x-1)
        maxx = min(x+2, n)
        miny = max(0, y-1)
        maxy = min(y+2, n)
        xs = list(range(minx, maxx))
        ys = list(range(miny, maxy))

        return ([y for y, x in product(ys, xs)], [x for y, x in product(ys, xs)])

    def get_neighbors(self, classes, blocki):
        ''' blocki is y, x '''
        y, x = blocki
        n = classes.shape[0]
        up = (y+1, x)
        down = (y-1, x)
        left = (y, x-1)
        right = (y, x+1)
        res = []
        if y+1 < n:
            res.append(up)
        if y-1 >= 0:
            res.append(down)
        if x+1 < n:
            res.append(right)
        if x-1 >= 0:
            res.append(left)
        return ([y for y, _ in res], [x for _, x in res]), (up, down, left, right)

    def has_same_neighbor(self, classes, blocki):
        """ checks if block has neighbors of same color """
        neighbs, (up, down, left, right) = self.get_neighbors(classes, blocki)
        classi = classes[blocki]
        return np.sum(classes[neighbs] == classes[blocki]) > 0

    def snake(self, classes, blocki):
        neighbi = self.get_area_neighbors(classes, blocki)
        return np.sum(classes[neighbi] == classes[blocki]) <= 2

    def can_switch(self, classes, blocki, target, n):
        """ blocki """
        if classes[blocki] == target:
            return False
        source = classes[blocki]

        classes[blocki] = target
        for neighbi in self.get_area(classes, blocki):
            if not self.has_same_neighbor(classes, neighbi):
                classes[blocki] = source
                return False
        classes[blocki] = source

        if self.snake(classes, blocki):
            return False

        _, (up, down, left, right) = self.get_neighbors(classes, blocki)
        if not self.check_split(classes, blocki, target, (up, down, left, right)):
            return False
        if not self.check_split(classes, blocki, target, (left, right, up, down)):
            return False

        return True

    def check_split(self, classes, blocki, target, dirs):
        (opp1, opp2, side1, side2) = dirs
        try:
            if classes[opp1] != classes[opp2]:
                return True
            if classes[opp1] != classes[blocki]:
                return True
        except IndexError:
            return True
        try:
            if classes[side1] == classes[blocki]:
                return True
        except IndexError:
            pass

        try:
            if classes[side2] == classes[blocki]:
                return True
        except IndexError:
            pass
        return False

    def adjust_population_classes(self, classes, n, iters=100, verbose=False, plotting=False):
        last_target = -1
        last_source = -1
        for i in range(iters):
            class_pops, class_count = self.calc_class_pop(classes, n)
            centers = self.get_dist_centers(classes, n)
            for top_class in np.flip(np.argsort(class_pops)):
                if np.random.rand() < .1:  # debug
                    continue
                if top_class == last_source:
                    continue
                if class_pops[top_class] < np.mean(class_pops):
                    break
                top_blocks = self.get_blocks(classes, top_class)
                dists = np.zeros((len(top_blocks), n))
                for blocki, block in enumerate(top_blocks):
                    for centeri, center in enumerate(centers):
                        block_center = np.array(self.get_block_center(*block))
                        dists[blocki, centeri] = self.dist(
                            block_center, center)
                closest = np.argsort(dists, axis=None)
                closest = np.unravel_index(closest, dists.shape)
                flipped = False
                for blocki, target in zip(*closest):
                    blocki = tuple(np.flip(top_blocks[blocki]))
                    if class_pops[target] > np.mean(class_pops):
                        continue
                    if target == last_source:
                        continue
                    if self.can_switch(classes, blocki, target, n):
                        flipped = True
                        if verbose:
                            print(class_pops / (np.mean(class_pops)))
                            print(np.flip(blocki),
                                  classes[blocki], 'to', target)
                        last_source = classes[blocki]
                        classes[blocki] = target
                        last_target = target
                        # print(class_pops)
                        break
                if flipped:
                    break
            if plotting and i % max(1, iters//10) == 0:
                print('plotting')
                self.plot_district_even(district=(classes, n, centers))
                plt.show()

        centers = self.get_dist_centers(classes, n)
        return classes, centers

    def district_even(self, n=None):
        """
        Returns tuple (row by col classes matrix of class, n, *values special
        to algorithm). In this case, special values is the kmeans centers.
        """
        manual = False
        classes, n, centers = self.get_kmeans_classes(n, manual=manual)
        if not manual:
            classes, centers = self.adjust_population_classes(classes, n)
        return (classes, n, centers)

    def district_pack(self, n):
        pass

    def district_crack(self, n):
        pass

    def get_proportion_z(self, class_ind_prop=0):
        # gets density of class class_ind_prop, really should only be used for
        # 2 class systems
        res = np.zeros((self.block_width, self.block_width))
        for x in range(self.block_width):
            for y in range(self.block_width):
                prop = [0 for _ in range(Person.num_classes)]
                for person in self.get_block_xy(x, y).people:
                    prop[person.class_ind] += 1
                if sum(prop) > 0:
                    res[y, x] = prop[class_ind_prop] / sum(prop)
                else:
                    res[y, x] = 0.5
        return res

    def plot_blocks(self):
        z = self.get_proportion_z()
        fig = plt.figure()
        plt.title('Block Preference Proportion')
        plt.pcolor(z, cmap='RdBu', vmin=0, vmax=1)
        tick_locs = range(self.block_width)
        # tick_labels = range(self.block_width)
        # plt.xticks(tick_locs, tick_labels, ha='left')
        plt.xticks(tick_locs, ha='left')
        plt.yticks(tick_locs, va='bottom')
        cbar = plt.colorbar()
        tick_labels = cbar.ax.get_yaxis().get_ticklabels()
        tick_labels[0] = Person.class_names[1]
        tick_labels[-1] = Person.class_names[0]
        cbar.ax.get_yaxis().set_ticklabels(tick_labels)
        return fig

    def get_class_xy(self):
        """
        Returns (
            ([x locs, ...], [y locs, ...]),
            ... for each class
        )

        Can be graphed with:
            import matplotlib.pyplot as plt

            state.plot_people()
            plt.show()

        """
        class_xy_pairs = tuple([([], []) for _ in range(Person.num_classes)])
        for (x, y), person in self.people.items():
            class_xy_pairs[person.class_ind][0].append(x)
            class_xy_pairs[person.class_ind][1].append(y)
        return class_xy_pairs

    def plot_people(self, marker=None):
        """
        if colors is specified, make sure it has enough colors for the
        classes (does support specifying more colors)

        marker should be '.' (small dot), ',' (pixel) or 'o' (large dot)
        """
        if marker is None:
            if self.num_people < 100:
                marker = 'o'
            elif self.num_people < 10000:
                marker = '.'
            else:
                marker = ','
        fig = plt.figure()
        values = self.get_class_xy()
        colors = 'brg'
        for i, ((X, Y), color) in enumerate(zip(values, colors)):
            class_name = Person.get_class_name(i)
            plt.plot(X, Y, f'{color}{marker}', label=class_name)

        plt.title("Population Class Distribution")
        tick_locs = np.linspace(0, 1, self.block_width+1)
        tick_labels = range(self.block_width)
        plt.xlim((0, 1))
        plt.xticks(tick_locs, tick_labels, ha='left')
        plt.ylim((0, 1))
        plt.yticks(tick_locs, tick_labels, va='bottom')
        plt.grid()
        plt.legend(loc='upper center', bbox_to_anchor=(
            0.5, -0.05), ncol=Person.num_classes)
        # plt.tight_layout()
        return fig
