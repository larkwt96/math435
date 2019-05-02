import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import product
import numpy as np
from sklearn.cluster import KMeans
from random import shuffle

# local imports
from block import Block
from person import Person


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

    def __init__(self, num_people=None, block_width=10):
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
        # create discrete colormap
        color_names = list(colors.BASE_COLORS.keys())
        color_names.remove('k')
        color_names.remove('w')
        shuffle(color_names)
        cmap = colors.ListedColormap(color_names)
        norm = colors.BoundaryNorm(list(range(n+1)), cmap.N)

        #plt.imshow(classes, cmap=cmap, norm=norm, origin='lower')
        plt.imshow(classes/n, cmap='viridis', origin='lower')

        # draw gridlines
        plt.grid(which='major', axis='both',
                 linestyle='-', color='k', linewidth=1)
        tick_locs = np.arange(self.block_width) - .5
        plt.xticks(tick_locs, range(self.block_width), ha='left')
        plt.yticks(tick_locs, range(self.block_width), va='bottom')

    def plot_district_even(self, n=None):
        if n is None:
            n = self.get_seats()
        district = self.district_even(n)
        self.plot_district(district)
        centers = self.block_width*district[2]-.5
        plt.scatter(centers[:, 0], centers[:, 1], c='k', s=5)

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

    def get_kmeans_classes(self, n=None):
        """
        returns classes, num classes (n) and k means centers (centers), row
        by col of class
        """
        if n is None:
            n = self.get_seats()
        X = np.array([[x, y] for x, y in self.people.keys()])
        kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        block_centers = self.get_block_centers()
        old_shape = block_centers.shape
        classes = kmeans.predict(block_centers.reshape(-1, 2))
        classes = classes.reshape(*old_shape[:-1])
        return classes, n, centers

    def adjust_population_classes(self, classes, n):
        reached_goal = False
        while not reached_goal:
            class_pops, class_count = self.calc_class_pop(classes, n)
            reached_goal = True
        print(class_pops, self.num_people/n)
        print(class_count)
        return classes

    def district_even(self, n=None):
        """
        Returns tuple (row by col classes matrix of class, n, *values special
        to algorithm). In this case, special values is the kmeans centers.
        """
        classes, n, centers = self.get_kmeans_classes(n)
        classes = self.adjust_population_classes(classes, n)
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
        #tick_labels = range(self.block_width)
        #plt.xticks(tick_locs, tick_labels, ha='left')
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
